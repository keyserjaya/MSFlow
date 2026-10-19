using System;
using UnityEngine;

namespace MSFlow.MotionBridge
{
    /// <summary>
    /// Plays a generated <see cref="MsflowMotion"/> on this character by
    /// writing bone local rotations in LateUpdate (after the Animator
    /// evaluates), blended by a weight so playback fades in from — and fades
    /// out to — whatever the Animator is currently doing. While the weight is
    /// 1 the generated pose fully overrides the Animator output; no
    /// AnimationClip is created, so state-machine integration (crossfade or
    /// queueing) is out of scope.
    /// </summary>
    [AddComponentMenu("MSFlow/MSFlow Playback Driver")]
    [DisallowMultipleComponent]
    public sealed class MsflowPlaybackDriver : MonoBehaviour
    {
        [Tooltip("Per source-joint rotation gates (index = joint id: 0 pelvis … 21 right_wrist). " +
                 "All bones are driven by default; untick a joint to leave it to the Animator.")]
        public bool[] applyRot = new bool[0];

        public MsflowRootMotionMode RootMotion = MsflowRootMotionMode.Hips;

        [Tooltip("MatchSourcePose: bones track the generated motion exactly — works with any " +
                 "rig rest (T-Pose, A-Pose, custom). RestAnchored: start from the rig's " +
                 "captured stance and apply the motion as a delta.")]
        public MsflowPoseTransferMode TransferMode = MsflowPoseTransferMode.MatchSourcePose;

        [Tooltip("Anchor the motion's start facing to the rig's rest facing (generated clips " +
                 "can face ~180 deg backward). Internal turns are preserved.")]
        public bool AlignFacing = true;

        [Range(0f, 1f)]
        [Tooltip("Flatten the spine zigzag (equalizes the relative bends of spine1/2/3 while " +
                 "keeping the overall torso lean). 0 = source-faithful, 1 = fully flattened.")]
        public float SpineSmoothing = 0f;

        public bool MatchLegScale = true;

        [Min(0f)]
        [Tooltip("Seconds to blend in and out of the generated pose.")]
        public float FadeSeconds = 0.25f;

        [Tooltip("Restart the motion when it finishes instead of fading out.")]
        public bool Loop = false;

        public bool IsPlaying => _state != PlaybackState.Idle;
        public MsflowMotion CurrentMotion => _motion;
        public float PlaybackWeight => _weight;
        public float PlaybackTime => _time;

        /// <summary>Raised when playback fully faded out (or was force-stopped).</summary>
        public event Action<MsflowPlaybackDriver> PlaybackEnded;

        private enum PlaybackState
        {
            Idle,
            FadeIn,
            Playing,
            FadeOut,
        }

        private MsflowMotion _motion;
        private MsflowRetargetEngine _engine;
        private PlaybackState _state = PlaybackState.Idle;
        private float _weight;
        private float _time;
        private Transform _hipsBone;
        private float _lastAppliedTime;
        private bool _hasAppliedTime;

        /// <summary>Start playback of a generated motion (restarts if playing).</summary>
        public void Play(MsflowMotion motion)
        {
            if (motion == null)
            {
                throw new ArgumentNullException(nameof(motion));
            }
            if (_state != PlaybackState.Idle)
            {
                ForceStop();
            }
            var animator = GetComponent<Animator>();
            if (animator == null)
            {
                animator = GetComponentInChildren<Animator>();
            }
            if (animator == null)
            {
                throw new InvalidOperationException(
                    "no Animator found on or under the playback driver's object");
            }
            var mapping = MsflowBoneMapper.Resolve(animator);
            foreach (var warning in mapping.Warnings)
            {
                Debug.Log($"[msflow] {warning}", this);
            }
            if (applyRot == null || applyRot.Length != MsflowSkeleton.JointCount)
            {
                applyRot = new bool[MsflowSkeleton.JointCount];
                for (var joint = 0; joint < applyRot.Length; joint++)
                {
                    applyRot[joint] = true;
                }
            }

            _engine = new MsflowRetargetEngine(
                motion, transform, mapping.Bones, mapping.Driven, RootMotion, MatchLegScale,
                TransferMode, AlignFacing, SpineSmoothing);
            _motion = motion;
            _hipsBone = mapping.Bones[MsflowSkeleton.Pelvis];
            _time = 0f;
            _weight = 0f;
            _hasAppliedTime = false;
            _lastAppliedTime = 0f;
            _state = PlaybackState.FadeIn;
        }

        /// <summary>Freeze the pose and fade back to the Animator.</summary>
        public void Stop()
        {
            if (_state == PlaybackState.FadeIn || _state == PlaybackState.Playing)
            {
                _state = PlaybackState.FadeOut;
            }
        }

        /// <summary>Drop the generated pose immediately (no fade).</summary>
        public void ForceStop()
        {
            if (_state == PlaybackState.Idle)
            {
                return;
            }
            EndPlayback();
        }

        /// <summary>
        /// The source joint position at <paramref name="time"/> in world space,
        /// anchored at the character's live hips position (scale-matched) — for
        /// debug overlays comparing the generated motion against the posed rig.
        /// </summary>
        public bool TryGetSourcePosition(int joint, float time, out Vector3 world)
        {
            if (_engine == null || _motion == null || _hipsBone == null)
            {
                world = Vector3.zero;
                return false;
            }
            var offset = _engine.SourceRootLocalPosition(time, joint) -
                         _engine.SourceRootLocalPosition(time, MsflowSkeleton.Pelvis);
            world = _hipsBone.position + _engine.RootRotation * (offset * _engine.ScaleFactor);
            return true;
        }

        private void LateUpdate()
        {
            if (_state == PlaybackState.Idle || _engine == null || _motion == null)
            {
                return;
            }
            var dt = Time.deltaTime;
            switch (_state)
            {
                case PlaybackState.FadeIn:
                    _time += dt;
                    _weight = Mathf.Min(1f, _weight + FadeRate(dt));
                    if (_weight >= 1f)
                    {
                        _state = PlaybackState.Playing;
                    }
                    break;
                case PlaybackState.Playing:
                    _time += dt;
                    if (_time >= _motion.DurationSec)
                    {
                        if (Loop)
                        {
                            _time %= _motion.DurationSec;
                        }
                        else
                        {
                            _time = _motion.DurationSec;
                            _state = PlaybackState.FadeOut;
                        }
                    }
                    break;
                case PlaybackState.FadeOut:
                    _weight = Mathf.Max(0f, _weight - FadeRate(dt));
                    if (_weight <= 0f)
                    {
                        EndPlayback();
                        return;
                    }
                    break;
            }

            var pose = _engine.Evaluate(_time);
            Apply(pose, _weight);
        }

        private void Apply(MsflowPose pose, float weight)
        {
            for (var i = 0; i < pose.Bones.Count; i++)
            {
                var target = pose.Bones[i];
                if (target.Joint < applyRot.Length && !applyRot[target.Joint])
                {
                    continue;
                }

                var bone = target.Bone;
                var worldTarget = _engine.RootRotation * target.WorldRotationRootLocal;
                var parent = bone.parent;
                var parentWorld = parent != null ? parent.rotation : Quaternion.identity;
                var local = Quaternion.Inverse(parentWorld) * worldTarget;
                bone.localRotation = Quaternion.Slerp(bone.localRotation, local, weight);
            }

            ApplyMovementDelta(weight);
        }

        /// <summary>
        /// Movement as incremental localPosition deltas (never absolute), so
        /// external locomotion keeps working during playback: each frame adds
        /// the source pelvis displacement since the last applied frame, scaled
        /// by leg ratio and the fade weight.
        /// </summary>
        private void ApplyMovementDelta(float weight)
        {
            if (_engine == null)
            {
                return;
            }
            if (!_hasAppliedTime)
            {
                _hasAppliedTime = true;
                _lastAppliedTime = _time;
                return;
            }
            if (_time <= _lastAppliedTime)
            {
                _lastAppliedTime = _time; // loop wrap or restart: no jump delta
                return;
            }
            var delta = _engine.SourceRootLocalPosition(_time, MsflowSkeleton.Pelvis) -
                        _engine.SourceRootLocalPosition(_lastAppliedTime, MsflowSkeleton.Pelvis);
            _lastAppliedTime = _time;
            delta *= _engine.ScaleFactor;
            if (RootMotion == MsflowRootMotionMode.InPlace)
            {
                delta = new Vector3(0f, delta.y, 0f);
            }
            if (delta.sqrMagnitude < 1e-12f)
            {
                return;
            }
            var worldDelta = _engine.RootRotation * delta * weight;
            if (RootMotion == MsflowRootMotionMode.RootTransform)
            {
                transform.position += worldDelta;
                return;
            }
            var hips = _hipsBone;
            if (hips != null && hips.parent != null)
            {
                hips.localPosition += hips.parent.InverseTransformVector(worldDelta);
            }
        }

        private void EndPlayback()
        {
            _state = PlaybackState.Idle;
            _weight = 0f;
            _engine = null;
            _motion = null;
            PlaybackEnded?.Invoke(this);
        }

        private float FadeRate(float dt)
        {
            return FadeSeconds <= 0f ? 1f : dt / FadeSeconds;
        }
    }
}
