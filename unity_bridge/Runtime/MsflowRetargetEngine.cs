using System.Collections.Generic;
using UnityEngine;

namespace MSFlow.MotionBridge
{
    /// <summary>Root-motion behaviour for generated playback.</summary>
    public enum MsflowRootMotionMode
    {
        /// <summary>Strip horizontal drift, keep the vertical bob (on the hips bone).</summary>
        InPlace,
        /// <summary>Apply the leg-scale-matched pelvis translation to the hips bone.</summary>
        Hips,
        /// <summary>Apply the full pelvis translation to the character root transform.</summary>
        RootTransform,
    }

    /// <summary>
    /// How the source pose is mapped onto the rig. The rig's rest pose is
    /// measured from the transforms (T-Pose, A-Pose, or any custom stance —
    /// no configuration needed either way); this selects only the transfer
    /// style.
    /// </summary>
    public enum MsflowPoseTransferMode
    {
        /// <summary>
        /// Bones aim exactly at the generated motion's joint directions and
        /// twist with its anatomical references (position-faithful). Works
        /// with any rig rest: T-Pose, A-Pose, or custom. At the reference
        /// frame the rig takes the motion's own first pose rather than its
        /// rest.
        /// </summary>
        MatchSourcePose,

        /// <summary>
        /// Apply the motion as a rotation delta on the rig's captured rest
        /// stance, so playback starts exactly at the character's rest pose.
        /// Limb aiming is only approximate when the rig's rest differs a lot
        /// from the generator's stance (arms-down canonical).
        /// </summary>
        RestAnchored,
    }

    /// <summary>One driven bone's desired world orientation (in root-local space).</summary>
    public struct MsflowBoneTarget
    {
        public int Joint;
        public Transform Bone;
        public Quaternion WorldRotationRootLocal;
    }

    /// <summary>The evaluated pose of one playback frame.</summary>
    public sealed class MsflowPose
    {
        /// <summary>Driven bones, FK order (parents before children).</summary>
        public readonly List<MsflowBoneTarget> Bones = new List<MsflowBoneTarget>();

        /// <summary>Scaled pelvis translation delta for this frame (root-local).</summary>
        public Vector3 PelvisDeltaRootLocal;
    }

    /// <summary>
    /// Per-play retarget engine: captures the rig rest, then turns msflow
    /// positions into per-bone world orientations via the anatomical twist
    /// reference construction (MsflowRetarget / RotationBuilder).
    ///
    /// MatchSourcePose: world(j,t) is the rotation mapping the bone's local
    /// rest directions (u1, u2) onto the source's pose directions (w1, w2) —
    /// the bone's anatomy follows the generated motion exactly for any rig
    /// rest. RestAnchored: world(j,t) = worldMS(j,t) . worldMS(j,ref)^-1 .
    /// AuthoredWorld(j), i.e. the pose change applied on the rig's rest.
    ///
    /// All math happens in the character root's space, captured once at play
    /// start, so playback is character-relative and root motion does not
    /// re-anchor the source data.
    /// </summary>
    public sealed class MsflowRetargetEngine
    {
        public float ScaleFactor { get; }
        public IReadOnlyList<int> DrivenJoints => _driven;
        public MsflowMotion Motion { get; }
        public Transform Root => _root;
        public Quaternion RootRotation => _rootRotation;

        /// <summary>Rest world position of the hips bone, in root-local space.</summary>
        public Vector3 HipsRestRootLocal => _targetRestPositions[MsflowSkeleton.Pelvis] ?? Vector3.zero;

        private readonly Transform _root;
        private readonly Matrix4x4 _rootWorldToLocal;
        private readonly Matrix4x4 _rootLocalToWorld;
        private readonly Quaternion _rootRotation;
        private readonly MsflowRootMotionMode _mode;
        private readonly MsflowPoseTransferMode _transfer;
        private readonly bool _matchLegScale;
        private readonly float _spineSmoothing;

        private readonly int[] _successors;
        private readonly Vector3?[] _sourceRefPositions = new Vector3?[MsflowSkeleton.JointCount];
        private readonly Vector3?[] _targetRestPositions = new Vector3?[MsflowSkeleton.JointCount];

        private readonly RestFrameSet _rest = new RestFrameSet();
        private readonly bool[] _mapped = new bool[MsflowSkeleton.JointCount];

        private readonly RotationState _playbackState = new RotationState();
        private readonly RotationState _refState = new RotationState();
        private readonly Quaternion[] _worldMs = new Quaternion[MsflowSkeleton.JointCount];
        private readonly Quaternion[] _refWorldMs = new Quaternion[MsflowSkeleton.JointCount];
        private readonly Quaternion _facingYaw = Quaternion.identity;

        private readonly List<int> _driven = new List<int>();
        private readonly Dictionary<int, Transform> _boneByJoint = new Dictionary<int, Transform>();
        private readonly MsflowPose _pose = new MsflowPose();
        private readonly Vector3[] _framePositions = new Vector3[MsflowSkeleton.JointCount];
        private readonly Vector3[] _framePositionsRootLocal = new Vector3[MsflowSkeleton.JointCount];

        public MsflowRetargetEngine(
            MsflowMotion motion,
            Transform root,
            Transform[] boneByJoint,
            bool[] drivenJoints,
            MsflowRootMotionMode mode = MsflowRootMotionMode.Hips,
            bool matchLegScale = true,
            MsflowPoseTransferMode transfer = MsflowPoseTransferMode.MatchSourcePose,
            bool alignFacing = true,
            float spineSmoothing = 0f)
        {
            Motion = motion;
            _root = root;
            _mode = mode;
            _transfer = transfer;
            _spineSmoothing = Mathf.Clamp01(spineSmoothing);
            _matchLegScale = matchLegScale;
            _rootWorldToLocal = root.worldToLocalMatrix;
            _rootLocalToWorld = root.localToWorldMatrix;
            _rootRotation = root.rotation;
            _successors = MsflowSkeleton.ChainSuccessors();

            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                var bone = boneByJoint != null && joint < boneByJoint.Length ? boneByJoint[joint] : null;
                if (bone != null)
                {
                    _boneByJoint[joint] = bone;
                    _mapped[joint] = true;
                    _targetRestPositions[joint] = _rootWorldToLocal.MultiplyPoint3x4(bone.position);
                    _rest.AuthoredWorld[joint] = bone.rotation;
                }
                if (drivenJoints != null && joint < drivenJoints.Length && drivenJoints[joint] && bone != null)
                {
                    _driven.Add(joint);
                }
            }

            // source reference frame = first motion frame, mapped into root
            // space (data convention -> Unity, see FromDataConvention)
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                var world = MsflowSkeleton.FromDataConvention(motion.GetJoint(0, joint));
                _sourceRefPositions[joint] = _rootWorldToLocal.MultiplyPoint3x4(world);
            }

            // facing alignment (bvh_to_vrm's Qw/ALIGN_TO_VRM_FACING): generated
            // clips can face ~180 deg from the rig's rest facing; anchor the
            // start facing with one constant yaw, keeping internal turns.
            if (alignFacing)
            {
                var facingYaw = ComputeFacingYaw(_sourceRefPositions, _targetRestPositions);
                _facingYaw = Quaternion.AngleAxis(facingYaw, Vector3.up);
                for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
                {
                    if (_sourceRefPositions[joint].HasValue)
                    {
                        _sourceRefPositions[joint] = _facingYaw * _sourceRefPositions[joint].Value;
                    }
                }
            }

            BuildRestFrames();
            ScaleFactor = _matchLegScale
                ? MsflowRetarget.LegScale(_sourceRefPositions, _targetRestPositions)
                : 1f;

            // FK order: sort driven joints by their rig hierarchy depth
            _driven.Sort((a, b) => HierarchyDepth(_boneByJoint[a]).CompareTo(HierarchyDepth(_boneByJoint[b])));

            // reference-frame world rotations (RestAnchored deltas)
            CopyPositions(_sourceRefPositions, _framePositionsRootLocal);
            RotationBuilder.Build(
                _framePositionsRootLocal, _successors, _rest, _mapped, _refState, _refWorldMs);
        }

        private void BuildRestFrames()
        {
            // character forward/back from the pelvis anatomy (constant twist priors)
            var p = _targetRestPositions;
            var forwardWorld = Vector3.zero;
            if (p[MsflowSkeleton.Pelvis].HasValue && p[MsflowSkeleton.LeftHip].HasValue &&
                p[MsflowSkeleton.RightHip].HasValue && p[MsflowSkeleton.Spine1].HasValue)
            {
                var hipLine = p[MsflowSkeleton.RightHip].Value - p[MsflowSkeleton.LeftHip].Value;
                var trunk = p[MsflowSkeleton.Spine1].Value - p[MsflowSkeleton.Pelvis].Value;
                var fwd = Vector3.Cross(hipLine, trunk);
                if (fwd.magnitude >= MsflowRetarget.Eps)
                {
                    forwardWorld = fwd.normalized;
                    _rest.ThighFallbackAxisLocal =
                        Quaternion.Inverse(_rest.AuthoredWorld[MsflowSkeleton.Pelvis]) * -forwardWorld;
                    _rest.HasThighFallback = true;
                }
            }

            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                _rest.HasU2[joint] = false;
                _rest.HasPrior[joint] = false;
                _rest.LeafParent[joint] = -1;
                _rest.LeafRelToParent[joint] = Quaternion.identity;
                if (!_mapped[joint])
                {
                    continue;
                }
                var auth = _rest.AuthoredWorld[joint];
                var invAuth = Quaternion.Inverse(auth);

                // u1: the bone's rest direction in bone-local space
                var tail = _successors[joint];
                var restDir = Vector3.zero;
                if (tail >= 0 && _targetRestPositions[tail].HasValue && _targetRestPositions[joint].HasValue)
                {
                    restDir = _targetRestPositions[tail].Value - _targetRestPositions[joint].Value;
                }
                if (restDir.magnitude < MsflowRetarget.Eps)
                {
                    // fall back to the bone's own rest child offset
                    var bone = _boneByJoint[joint];
                    if (bone.childCount > 0)
                    {
                        restDir = bone.GetChild(0).position - bone.position;
                    }
                }
                if (restDir.magnitude >= MsflowRetarget.Eps)
                {
                    _rest.U1[joint] = invAuth * restDir.normalized;
                }
                else
                {
                    _rest.U1[joint] = Vector3.up;
                }

                // u2: the structural twist analog from the target rest geometry
                // first (keeps the rest pose exact); the constant anatomical
                // prior (arms: forward elbow plane, thighs: backward knee
                // plane — bvh_to_vrm's rules) fills in only where straight-limb
                // rests make the structural analog degenerate.
                {
                    var pair0 = MsflowRetarget.TwistPairs[joint, 0];
                    var pair1 = MsflowRetarget.TwistPairs[joint, 1];
                    if (pair0 >= 0 && pair1 >= 0 &&
                        _targetRestPositions[pair0].HasValue && _targetRestPositions[pair1].HasValue)
                    {
                        var v2 = _targetRestPositions[pair1].Value - _targetRestPositions[pair0].Value;
                        _rest.HasU2[joint] = MsflowRetarget.TryOrtho(
                            invAuth * v2, _rest.U1[joint], out var u2);
                        if (_rest.HasU2[joint])
                        {
                            _rest.U2[joint] = u2;
                        }
                    }
                }
                if (!_rest.HasU2[joint] && forwardWorld.magnitude >= MsflowRetarget.Eps &&
                    (joint == MsflowSkeleton.LeftHip || joint == MsflowSkeleton.RightHip ||
                     joint == MsflowSkeleton.LeftShoulder || joint == MsflowSkeleton.RightShoulder ||
                     joint == MsflowSkeleton.LeftElbow || joint == MsflowSkeleton.RightElbow))
                {
                    var priorWorld = (joint == MsflowSkeleton.LeftHip || joint == MsflowSkeleton.RightHip)
                        ? -forwardWorld
                        : forwardWorld;
                    _rest.HasPrior[joint] = MsflowRetarget.TryOrtho(
                        invAuth * priorWorld, _rest.U1[joint], out var prior);
                    if (_rest.HasPrior[joint])
                    {
                        _rest.PriorAxisLocal[joint] = prior;
                        _rest.U2[joint] = prior;
                        _rest.HasU2[joint] = true;
                    }
                }

                // leaves inherit their rest-relative pose inside the parent frame
                if (!MsflowRetarget.HasDirectionRef(joint))
                {
                    var parent = MsflowSkeleton.Parents[joint];
                    while (parent >= 0 && !_mapped[parent])
                    {
                        parent = MsflowSkeleton.Parents[parent];
                    }
                    if (parent >= 0)
                    {
                        _rest.LeafParent[joint] = parent;
                        _rest.LeafRelToParent[joint] =
                            Quaternion.Inverse(_rest.AuthoredWorld[parent]) * auth;
                    }
                }
            }
        }

        private static void CopyPositions(Vector3?[] source, Vector3[] destination)
        {
            for (var i = 0; i < source.Length; i++)
            {
                destination[i] = source[i] ?? Vector3.zero;
            }
        }

        /// <summary>
        /// Yaw (degrees about root-local up) taking the source's reference
        /// facing onto the rig's rest facing — both from hip-line x trunk.
        /// </summary>
        private static float ComputeFacingYaw(
            Vector3?[] sourceRef, Vector3?[] targetRest)
        {
            var facingSource = FacingOf(sourceRef);
            var facingTarget = FacingOf(targetRest);
            var yawSource = Mathf.Atan2(facingSource.x, facingSource.z) * Mathf.Rad2Deg;
            var yawTarget = Mathf.Atan2(facingTarget.x, facingTarget.z) * Mathf.Rad2Deg;
            return Mathf.DeltaAngle(yawSource, yawTarget);
        }

        private static Vector3 FacingOf(Vector3?[] positions)
        {
            if (positions[MsflowSkeleton.Pelvis].HasValue &&
                positions[MsflowSkeleton.LeftHip].HasValue &&
                positions[MsflowSkeleton.RightHip].HasValue &&
                positions[MsflowSkeleton.Spine1].HasValue)
            {
                var hipLine = positions[MsflowSkeleton.RightHip].Value -
                              positions[MsflowSkeleton.LeftHip].Value;
                var trunk = positions[MsflowSkeleton.Spine1].Value -
                            positions[MsflowSkeleton.Pelvis].Value;
                var facing = Vector3.Cross(hipLine, trunk);
                if (facing.magnitude >= MsflowRetarget.Eps)
                {
                    return facing.normalized;
                }
            }
            return Vector3.forward;
        }

        private static int HierarchyDepth(Transform transform)
        {
            var depth = 0;
            var walker = transform.parent;
            while (walker != null)
            {
                depth++;
                walker = walker.parent;
            }
            return depth;
        }

        /// <summary>Evaluate the pose at <paramref name="time"/> (seconds).</summary>
        public MsflowPose Evaluate(float time)
        {
            var motion = Motion;
            var frameTime = motion.ClampTime(time) * motion.Fps;
            var i0 = Mathf.Min((int)frameTime, motion.FrameCount - 1);
            var i1 = Mathf.Min(i0 + 1, motion.FrameCount - 1);
            var frac = Mathf.Clamp01(frameTime - i0);

            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                var world = Vector3.LerpUnclamped(
                    motion.GetJoint(i0, joint), motion.GetJoint(i1, joint), frac);
                _framePositions[joint] = world;
                _framePositionsRootLocal[joint] = _facingYaw * _rootWorldToLocal.MultiplyPoint3x4(
                    MsflowSkeleton.FromDataConvention(world));
            }

            RotationBuilder.Build(
                _framePositionsRootLocal, _successors, _rest, _mapped, _playbackState, _worldMs);
            if (_spineSmoothing > 0f)
            {
                SmoothSpineChain();
            }

            var pelvis = _framePositionsRootLocal[MsflowSkeleton.Pelvis];
            var pelvisRef = _sourceRefPositions[MsflowSkeleton.Pelvis].Value;
            var delta = (pelvis - pelvisRef) * ScaleFactor;
            if (_mode == MsflowRootMotionMode.InPlace)
            {
                delta = new Vector3(0f, delta.y, 0f);
            }

            _pose.Bones.Clear();
            for (var index = 0; index < _driven.Count; index++)
            {
                var joint = _driven[index];
                Quaternion worldRootLocal;
                if (_transfer == MsflowPoseTransferMode.MatchSourcePose)
                {
                    worldRootLocal = _worldMs[joint];
                }
                else
                {
                    worldRootLocal = _worldMs[joint] * Quaternion.Inverse(_refWorldMs[joint]) *
                                     _rest.AuthoredWorld[joint];
                }
                _pose.Bones.Add(new MsflowBoneTarget
                {
                    Joint = joint,
                    Bone = _boneByJoint[joint],
                    WorldRotationRootLocal = worldRootLocal,
                });
            }
            _pose.PelvisDeltaRootLocal = delta;
            return _pose;
        }

        /// <summary>World-space hips target position for one frame.</summary>
        public Vector3 HipsWorldTarget(Vector3 pelvisDeltaRootLocal)
        {
            return _rootLocalToWorld.MultiplyPoint3x4(pelvisDeltaRootLocal + HipsRestRootLocal);
        }

        /// <summary>
        /// The aligned source joint position at <paramref name="time"/> in the
        /// character root's local space (translation-invariant; differences of
        /// two joints/times give motion deltas).
        /// </summary>
        public Vector3 SourceRootLocalPosition(float time, int joint)
        {
            var motion = Motion;
            var frameTime = motion.ClampTime(time) * motion.Fps;
            var i0 = Mathf.Min((int)frameTime, motion.FrameCount - 1);
            var i1 = Mathf.Min(i0 + 1, motion.FrameCount - 1);
            var frac = Mathf.Clamp01(frameTime - i0);
            var world = Vector3.LerpUnclamped(
                motion.GetJoint(i0, joint), motion.GetJoint(i1, joint), frac);
            return _facingYaw * _rootWorldToLocal.MultiplyPoint3x4(
                MsflowSkeleton.FromDataConvention(world));
        }

        /// <summary>
        /// The aligned source joint position at <paramref name="time"/> in
        /// world space (at the capture-time root placement) — reference math
        /// only; playback uses deltas, see SourceRootLocalPosition.
        /// </summary>
        public Vector3 SourceWorldPosition(float time, int joint)
        {
            return _rootLocalToWorld.MultiplyPoint3x4(SourceRootLocalPosition(time, joint));
        }

        /// <summary>
        /// Flatten the spine zigzag: equalize the relative bends of the
        /// spine1/2/3 chain while preserving the overall torso lean (the
        /// generated clips can crunch adjacent segments in opposite
        /// directions). Twist is kept from the original frames.
        /// </summary>
        private void SmoothSpineChain()
        {
            var a = _framePositionsRootLocal[MsflowSkeleton.Spine1];
            var b = _framePositionsRootLocal[MsflowSkeleton.Spine2];
            var c = _framePositionsRootLocal[MsflowSkeleton.Spine3];
            var d = _framePositionsRootLocal[MsflowSkeleton.Neck];
            var d1 = b - a;
            var d2 = c - b;
            var d3 = d - c;
            if (d1.magnitude < MsflowRetarget.Eps || d2.magnitude < MsflowRetarget.Eps ||
                d3.magnitude < MsflowRetarget.Eps)
            {
                return;
            }
            d1.Normalize();
            d2.Normalize();
            d3.Normalize();
            var bend12 = Quaternion.FromToRotation(d1, d2);
            var bend23 = Quaternion.FromToRotation(d2, d3);
            var average = Quaternion.Slerp(bend12, bend23, 0.5f);
            var d2Target = average * d1;
            var d3Target = average * d2Target;
            SmoothAim(MsflowSkeleton.Spine2, d2, d2Target);
            SmoothAim(MsflowSkeleton.Spine3, d3, d3Target);
        }

        private void SmoothAim(int joint, Vector3 current, Vector3 target)
        {
            var aim = Vector3.Slerp(current, target, _spineSmoothing).normalized;
            if (!_rest.HasU2[joint])
            {
                return;
            }
            var twist = _worldMs[joint] * _rest.U2[joint];
            if (!MsflowRetarget.TryOrtho(twist, aim, out var w2))
            {
                return;
            }
            if (MsflowRetarget.TryFrameFromRefs(_rest.U1[joint], _rest.U2[joint], aim, w2, out var world))
            {
                _worldMs[joint] = world;
            }
        }
    }
}
