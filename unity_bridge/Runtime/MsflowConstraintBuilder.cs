using System;
using System.Collections.Generic;
using UnityEngine;

namespace MSFlow.MotionBridge
{
    /// <summary>
    /// Converts Unity-side bone intent into sparse-control wire entries (the
    /// webUI spec-loader schema, <see cref="MsflowControl"/>) so callers can
    /// author generation constraints without knowing the 22-joint layout,
    /// the data convention (X-mirrored relative to Unity), or the canonical
    /// scale:
    ///
    ///     var b = MsflowConstraintBuilder.ForCharacter(animator);
    ///     b.PinBone(HumanBodyBones.LeftFoot, 0.5f, worldPosition);
    ///     b.PinCurrentPoseAtStart();
    ///     var controls = b.Build();
    ///     await request.GenerateAsync(prompt, 3f, controls);
    ///
    /// Pins are captured at call time (deterministic) and keyed by
    /// (joint, frame fraction rounded to 1e-3) — a later pin on the same
    /// key replaces the earlier one. Constraints pin joint *positions* only:
    /// the wire protocol cannot express rotation constraints.
    ///
    /// Conversion per pin: <c>data = FromDataConvention(rootLocal) / scale</c>
    /// where <c>rootLocal = root.worldToLocalMatrix * worldPos</c> (so pins
    /// ride along with character movement) and <c>scale</c> is the character
    /// leg span over the canonical 0.88 m (the engine plays the result back
    /// at that same ratio). The X-flip involution makes pins round-trip
    /// exactly through the engine's output mapping.
    /// </summary>
    public sealed class MsflowConstraintBuilder
    {
        /// <summary>Mean hip→ankle span of the canonical figure (metres).</summary>
        public const float CanonicalLegSpan = 0.88f;

        /// <summary>Frame rate of msflow-motion artifacts (frames per second).</summary>
        public const float MotionFrameRate = 20f;

        /// <summary>
        /// Clip length (frames) that <see cref="PinBoneThroughout"/> expands
        /// against. The GenerateAsync builder overloads sync it from the
        /// requested duration; set it yourself before Build() when sending
        /// the controls by hand (120 = the webUI spec default).
        /// </summary>
        public int FrameCount { get; set; } = DefaultFrameCount;

        public const int DefaultFrameCount = 120;

        /// <summary>Send pins in character/canonical scale (default), or raw.</summary>
        public bool NormalizeScale = true;

        /// <summary>Optional wire `role` label stamped on every emitted control.</summary>
        public string Role = "";

        /// <summary>Character leg span ÷ canonical leg span (1 when not normalizing).</summary>
        public float ScaleFactor =>
            NormalizeScale && _legSpan > 1e-5f ? _legSpan / CanonicalLegSpan : 1f;

        /// <summary>The character root pins are expressed relative to.</summary>
        public Transform Root => _root;

        private static readonly string[] AxisNames = { "x", "y", "z" };

        private readonly Transform _root;
        private readonly Transform[] _bones;
        private readonly Animator _animator;
        private readonly float _legSpan;

        private struct Pin
        {
            public int Joint;
            public float Fraction;
            public Vector3 Value;
            public string Axes;
            public bool Throughout;
            public int Interval;

            public int FractionKey => (int)Math.Round(Fraction * 1000.0);
        }

        private readonly List<Pin> _pins = new List<Pin>();

        private MsflowConstraintBuilder(Transform root, Transform[] bones, Animator animator)
        {
            _root = root;
            _bones = bones;
            _animator = animator;
            _legSpan = MeasureLegSpan(bones);
        }

        /// <summary>Builder for a Humanoid character (root = the Animator's transform).</summary>
        public static MsflowConstraintBuilder ForCharacter(Animator animator)
        {
            if (animator == null)
            {
                throw new ArgumentNullException(nameof(animator));
            }
            var mapping = MsflowBoneMapper.Resolve(animator);
            return new MsflowConstraintBuilder(animator.transform, mapping.Bones, animator);
        }

        /// <summary>
        /// Builder for a playback driver's character (root = the driver's
        /// transform, exactly the space playback re-anchors the motion in).
        /// </summary>
        public static MsflowConstraintBuilder ForDriver(MsflowPlaybackDriver driver)
        {
            if (driver == null)
            {
                throw new ArgumentNullException(nameof(driver));
            }
            var animator = driver.GetComponent<Animator>();
            if (animator == null)
            {
                animator = driver.GetComponentInChildren<Animator>();
            }
            if (animator == null)
            {
                throw new InvalidOperationException(
                    "no Animator found on or under the playback driver's object");
            }
            var mapping = MsflowBoneMapper.Resolve(animator);
            return new MsflowConstraintBuilder(driver.transform, mapping.Bones, animator);
        }

        /// <summary>
        /// Builder from an explicit root and 22-slot bone table (null = joint
        /// without a bone). Escape hatch for tests and custom rigs; bone names
        /// resolve against the canonical bone→joint table.
        /// </summary>
        public static MsflowConstraintBuilder ForRig(Transform root, Transform[] bonesByJoint)
        {
            if (root == null)
            {
                throw new ArgumentNullException(nameof(root));
            }
            if (bonesByJoint == null)
            {
                throw new ArgumentNullException(nameof(bonesByJoint));
            }
            if (bonesByJoint.Length != MsflowSkeleton.JointCount)
            {
                throw new ArgumentException(
                    $"bonesByJoint must have {MsflowSkeleton.JointCount} slots, got {bonesByJoint.Length}",
                    nameof(bonesByJoint));
            }
            return new MsflowConstraintBuilder(root, (Transform[])bonesByJoint.Clone(), null);
        }

        /// <summary>
        /// Pin one bone to a world position at a clip fraction (0 = first
        /// frame, 1 = last frame). The value is converted to data space at
        /// call time.
        /// </summary>
        public void PinBone(
            HumanBodyBones bone, float frameFraction, Vector3 worldPosition, string axes = "x,y,z")
        {
            var joint = ResolveBone(bone);
            PinJoint(joint, frameFraction, worldPosition, axes);
        }

        /// <summary>
        /// Pin one source joint (0=…21, see <see cref="MsflowSkeleton.JointNames"/>)
        /// to a world position at a clip fraction. Escape hatch for callers
        /// that already know the joint layout.
        /// </summary>
        public void PinJoint(
            int joint, float frameFraction, Vector3 worldPosition, string axes = "x,y,z")
        {
            if (joint < 0 || joint >= MsflowSkeleton.JointCount)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(joint), joint,
                    $"joint must be in [0, {MsflowSkeleton.JointCount}), see MsflowSkeleton.JointNames");
            }
            var fraction = ValidateFraction(frameFraction);
            var normalizedAxes = NormalizeAxes(axes);
            _pins.Add(new Pin
            {
                Joint = joint,
                Fraction = fraction,
                Value = ToDataValue(worldPosition),
                Axes = normalizedAxes,
            });
        }

        /// <summary>
        /// Hold the bone's current world position at every
        /// <paramref name="intervalFrames"/>-th frame of a
        /// <see cref="FrameCount"/>-frame clip (frames 0, interval, 2×interval, …).
        /// </summary>
        public void PinBoneThroughout(
            HumanBodyBones bone, int intervalFrames = 10, string axes = "x,y,z")
        {
            if (intervalFrames < 1)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(intervalFrames), intervalFrames, "intervalFrames must be >= 1");
            }
            var joint = ResolveBone(bone);
            var normalizedAxes = NormalizeAxes(axes);
            _pins.Add(new Pin
            {
                Joint = joint,
                Fraction = 0f,
                Value = ToDataValue(_bones[joint].position),
                Axes = normalizedAxes,
                Throughout = true,
                Interval = intervalFrames,
            });
        }

        /// <summary>
        /// Snapshot the live pose at a clip fraction: one pin per mapped
        /// joint (joints without a bone are skipped with one aggregated
        /// warning). Pass <paramref name="only"/> to restrict the snapshot to
        /// specific bones (each must exist on the character).
        /// </summary>
        public void PinPose(
            float frameFraction, string axes = "x,y,z", params HumanBodyBones[] only)
        {
            var fraction = ValidateFraction(frameFraction);
            var normalizedAxes = NormalizeAxes(axes);
            if (only == null || only.Length == 0)
            {
                var skipped = new List<string>();
                for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
                {
                    if (_bones[joint] == null)
                    {
                        skipped.Add(MsflowSkeleton.JointNames[joint]);
                        continue;
                    }
                    _pins.Add(new Pin
                    {
                        Joint = joint,
                        Fraction = fraction,
                        Value = ToDataValue(_bones[joint].position),
                        Axes = normalizedAxes,
                    });
                }
                if (skipped.Count > 0)
                {
                    Debug.LogWarning(
                        $"[msflow] PinPose skipped {skipped.Count} unmapped joint(s): " +
                        string.Join(", ", skipped));
                }
                return;
            }
            foreach (var bone in only)
            {
                var joint = ResolveBone(bone);
                _pins.Add(new Pin
                {
                    Joint = joint,
                    Fraction = fraction,
                    Value = ToDataValue(_bones[joint].position),
                    Axes = normalizedAxes,
                });
            }
        }

        /// <summary>Snapshot the live pose and pin it at the clip start (frame 0).</summary>
        public void PinCurrentPoseAtStart(string axes = "x,y,z")
        {
            PinPose(0f, axes);
        }

        /// <summary>Snapshot the live pose and pin it at both clip ends.</summary>
        public void PinCurrentPoseAtStartAndEnd(string axes = "x,y,z")
        {
            PinPose(0f, axes);
            PinPose(1f, axes);
        }

        /// <summary>Drop every pin captured so far.</summary>
        public void Clear()
        {
            _pins.Clear();
        }

        /// <summary>
        /// Emit the wire controls: one entry per (frame fraction, axes) group,
        /// joints ascending. Duplicates (same joint + fraction within 1e-3)
        /// collapse to the latest pin.
        /// </summary>
        public List<MsflowControl> Build()
        {
            var pins = new List<Pin>();
            var slotByKey = new Dictionary<long, int>();
            foreach (var pin in ExpandPins())
            {
                var key = PinKey(pin.Joint, pin.FractionKey);
                if (slotByKey.TryGetValue(key, out var slot))
                {
                    pins[slot] = pin; // later pin wins
                }
                else
                {
                    slotByKey[key] = pins.Count;
                    pins.Add(pin);
                }
            }
            pins.Sort(ComparePins);

            var controls = new List<MsflowControl>();
            var start = 0;
            while (start < pins.Count)
            {
                var end = start + 1;
                while (end < pins.Count &&
                       pins[end].FractionKey == pins[start].FractionKey &&
                       pins[end].Axes == pins[start].Axes)
                {
                    end++;
                }
                var joints = new int[end - start];
                var values = new float[end - start][];
                for (var i = start; i < end; i++)
                {
                    joints[i - start] = pins[i].Joint;
                    var value = pins[i].Value;
                    values[i - start] = new[] { value.x, value.y, value.z };
                }
                controls.Add(new MsflowControl
                {
                    FrameFraction = pins[start].Fraction,
                    Joints = joints,
                    Axes = pins[start].Axes,
                    Values = values,
                    Role = Role,
                });
                start = end;
            }
            return controls;
        }

        /// <summary>
        /// Merge two wire control lists: pins in <paramref name="overrides"/>
        /// win over same-joint pins at the same clip position in
        /// <paramref name="baseControls"/> (which are split per joint around
        /// the conflicts). Positions compare by frame_fraction (rounded to
        /// 1e-3); a frame pin at 0 equals a frame_fraction pin at 0.
        /// </summary>
        public static List<MsflowControl> MergeControls(
            IList<MsflowControl> baseControls, IList<MsflowControl> overrides)
        {
            var covered = new HashSet<long>();
            if (overrides != null)
            {
                foreach (var control in overrides)
                {
                    var key = PositionKey(control);
                    foreach (var joint in control.Joints ?? Array.Empty<int>())
                    {
                        covered.Add(PairKey(joint, key));
                    }
                }
            }

            var merged = new List<MsflowControl>();
            if (baseControls != null)
            {
                foreach (var control in baseControls)
                {
                    var joints = control.Joints ?? Array.Empty<int>();
                    var values = control.Values ?? Array.Empty<float[]>();
                    var keptJoints = new List<int>();
                    var keptValues = new List<float[]>();
                    var key = PositionKey(control);
                    for (var i = 0; i < joints.Length; i++)
                    {
                        if (covered.Contains(PairKey(joints[i], key)))
                        {
                            continue;
                        }
                        keptJoints.Add(joints[i]);
                        if (i < values.Length)
                        {
                            keptValues.Add(values[i]);
                        }
                    }
                    if (keptJoints.Count == 0)
                    {
                        continue;
                    }
                    merged.Add(new MsflowControl
                    {
                        Frame = control.Frame,
                        FrameFraction = control.FrameFraction,
                        Joints = keptJoints.ToArray(),
                        Axes = control.Axes,
                        Values = keptValues.ToArray(),
                        Stage = control.Stage,
                        Role = control.Role,
                    });
                }
            }
            if (overrides != null)
            {
                merged.AddRange(overrides);
            }
            return merged;
        }

        private IEnumerable<Pin> ExpandPins()
        {
            foreach (var pin in _pins)
            {
                if (!pin.Throughout)
                {
                    yield return pin;
                    continue;
                }
                var last = Math.Max(FrameCount - 1, 0);
                for (var frame = 0; frame <= last; frame += pin.Interval)
                {
                    yield return new Pin
                    {
                        Joint = pin.Joint,
                        Fraction = last > 0 ? (float)frame / last : 0f,
                        Value = pin.Value,
                        Axes = pin.Axes,
                    };
                }
            }
        }

        private static int ComparePins(Pin a, Pin b)
        {
            var byFraction = a.FractionKey.CompareTo(b.FractionKey);
            if (byFraction != 0)
            {
                return byFraction;
            }
            var byAxes = string.CompareOrdinal(a.Axes, b.Axes);
            if (byAxes != 0)
            {
                return byAxes;
            }
            return a.Joint.CompareTo(b.Joint);
        }

        private int ResolveBone(HumanBodyBones bone)
        {
            if (!MsflowBoneMapper.TryGetJointForBone(bone, out var joint))
            {
                throw new ArgumentException(
                    $"bone {bone} has no MSFlow source joint (fingers and eyes are not " +
                    $"expressible); supported bones: {SupportedBonesText()}",
                    nameof(bone));
            }
            if (_animator != null)
            {
                var transform = _animator.GetBoneTransform(bone);
                if (transform == null)
                {
                    throw new ArgumentException(
                        $"bone {bone} is not on this character's avatar; " +
                        $"supported bones: {SupportedBonesText()}",
                        nameof(bone));
                }
                if (_bones[joint] != transform)
                {
                    // spine slots fold onto the avatar's Spine/Chest/UpperChest
                    // bones; re-locate the slot that actually holds this bone
                    for (var j = 0; j < _bones.Length; j++)
                    {
                        if (_bones[j] == transform)
                        {
                            joint = j;
                            break;
                        }
                    }
                }
            }
            if (_bones[joint] == null)
            {
                throw new ArgumentException(
                    $"bone {bone} is not mapped on this character; " +
                    $"supported bones: {SupportedBonesText()}",
                    nameof(bone));
            }
            return joint;
        }

        private static string SupportedBonesText()
        {
            return string.Join(", ", MsflowBoneMapper.SupportedBones);
        }

        private Vector3 ToDataValue(Vector3 worldPosition)
        {
            var rootLocal = _root.worldToLocalMatrix.MultiplyPoint3x4(worldPosition);
            return MsflowSkeleton.FromDataConvention(rootLocal) / ScaleFactor;
        }

        private static float ValidateFraction(float frameFraction)
        {
            if (!(frameFraction >= 0f && frameFraction <= 1f))
            {
                throw new ArgumentOutOfRangeException(
                    nameof(frameFraction), frameFraction, "frame_fraction must be in [0, 1]");
            }
            return frameFraction;
        }

        private static string NormalizeAxes(string axes)
        {
            if (string.IsNullOrWhiteSpace(axes))
            {
                throw new ArgumentException(
                    "axes must list at least one of x,y,z", nameof(axes));
            }
            var seen = new bool[3];
            foreach (var raw in axes.Split(','))
            {
                var token = raw.Trim().ToLowerInvariant();
                int axis;
                if (token == "x")
                {
                    axis = 0;
                }
                else if (token == "y")
                {
                    axis = 1;
                }
                else if (token == "z")
                {
                    axis = 2;
                }
                else
                {
                    throw new ArgumentException(
                        $"unknown axis '{raw.Trim()}' (axes is a comma-separated subset of x,y,z)",
                        nameof(axes));
                }
                seen[axis] = true;
            }
            var picked = new List<string>(3);
            for (var axis = 0; axis < 3; axis++)
            {
                if (seen[axis])
                {
                    picked.Add(AxisNames[axis]);
                }
            }
            return string.Join(",", picked);
        }

        private static float MeasureLegSpan(Transform[] bones)
        {
            float Span(int hip, int ankle)
            {
                return bones[hip] != null && bones[ankle] != null
                    ? Vector3.Distance(bones[hip].position, bones[ankle].position)
                    : 0f;
            }
            return (Span(MsflowSkeleton.LeftHip, MsflowSkeleton.LeftAnkle) +
                    Span(MsflowSkeleton.RightHip, MsflowSkeleton.RightAnkle)) * 0.5f;
        }

        private static long PinKey(int joint, int fractionKey)
        {
            return ((long)joint << 20) | (uint)fractionKey;
        }

        private static long PairKey(int joint, int positionKey)
        {
            return PinKey(joint, positionKey);
        }

        /// <summary>
        /// Clip-position key of one wire control: frame_fraction rounded to
        /// 1e-3; frame pins in their own space (frame 0 == fraction 0).
        /// </summary>
        private static int PositionKey(MsflowControl control)
        {
            if (control.FrameFraction.HasValue)
            {
                return (int)Math.Round(control.FrameFraction.Value * 1000.0);
            }
            return control.Frame == 0 ? 0 : 1000 + control.Frame.Value;
        }
    }
}
