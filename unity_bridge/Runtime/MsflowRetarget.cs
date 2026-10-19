using System.Collections.Generic;
using UnityEngine;

namespace MSFlow.MotionBridge
{
    /// <summary>
    /// Position-based orientation reconstruction (port of the proven design in
    /// BlenderScripts/bvh_to_vrm.py, adapted to the 22-joint HumanML3D
    /// skeleton). For every bone the pose rotation is the one that maps:
    ///   - the bone's rest direction onto the source segment direction w1, and
    ///   - the bone's rest twist reference u2 onto the source twist reference w2
    /// (a stable anatomical second direction: shoulder line for spine/neck,
    /// elbow/knee planes for limbs, hip line for the pelvis). Two-vector
    /// alignment beats roll-carrier chains, which flip when candidates go
    /// near-parallel. Bones without a direction (head, hands, toes here) keep
    /// their rest-relative pose inside the parent's animated frame.
    /// </summary>
    public static class MsflowRetarget
    {
        public const float Eps = 1e-8f;
        private const float Degenerate = 1e-6f;

        /// <summary>
        /// Roll-reference damping toward the previous frame (0 = raw refs,
        /// 1 = no change ever). Kills roll flip-flops without touching the
        /// aim direction. Settable so tests can pin exact roll.
        /// </summary>
        public static float TwistSmoothing = 0.5f;

        /// <summary>
        /// Per joint (c, d): the twist reference pair, v2 = pos[d] - pos[c],
        /// orthogonalized against the bone direction. (-1, -1) = leaf bones
        /// without a direction reference (they inherit the parent frame).
        /// Orders matter and must match on the target side.
        /// </summary>
        public static readonly int[,] TwistPairs =
        {
            { 2, 1 },    // 0  pelvis:   hip line (right -> left)
            { 4, 7 },    // 1  left_hip:  shin (knee -> ankle)
            { 5, 8 },    // 2  right_hip: shin
            { 13, 14 },  // 3  spine1:   collar line (left -> right)
            { 7, 10 },   // 4  left_knee: foot
            { 8, 11 },   // 5  right_knee: foot
            { 13, 14 },  // 6  spine2:   collar line
            { 7, 4 },    // 7  left_ankle: ankle -> knee (up the leg)
            { 8, 5 },    // 8  right_ankle
            { 13, 14 },  // 9  spine3:   collar line
            { -1, -1 },  // 10 left_foot: leaf
            { -1, -1 },  // 11 right_foot: leaf
            { 13, 14 },  // 12 neck:     collar line
            { 0, 12 },   // 13 left_collar: torso up (hips -> neck)
            { 0, 12 },   // 14 right_collar
            { -1, -1 },  // 15 head: leaf (inherits the neck)
            { 18, 20 },  // 16 left_shoulder: forearm (elbow bend plane)
            { 19, 21 },  // 17 right_shoulder
            { 16, 18 },  // 18 left_elbow: upper arm (elbow bend plane)
            { 17, 19 },  // 19 right_elbow
            { -1, -1 },  // 20 left_wrist: leaf (no fingers in the source)
            { -1, -1 },  // 21 right_wrist
        };

        /// <summary>Joints with a direction reference (non-leaf).</summary>
        public static bool HasDirectionRef(int joint)
        {
            return joint != MsflowSkeleton.LeftFoot && joint != MsflowSkeleton.RightFoot &&
                   joint != MsflowSkeleton.Head &&
                   joint != MsflowSkeleton.LeftWrist && joint != MsflowSkeleton.RightWrist;
        }

        /// <summary>
        /// Component of <paramref name="v"/> perpendicular to <paramref name="axis"/>.
        /// Requires a meaningful bend angle (relative threshold): near-straight
        /// limbs carry no usable twist signal, and epsilon-level ortho
        /// components are noise that flips the roll.
        /// </summary>
        public static bool TryOrtho(Vector3 v, Vector3 axis, out Vector3 result)
        {
            var d = v - axis * Vector3.Dot(v, axis);
            var threshold = Mathf.Max(Degenerate, 0.2f * v.magnitude);
            if (d.magnitude < threshold)
            {
                result = Vector3.zero;
                return false;
            }
            result = d.normalized;
            return true;
        }

        /// <summary>
        /// Rotation mapping local frame (u1, u2, u1xu2) onto world frame
        /// (w1, w2, w1xw2): u1->w1 and u2->w2 (bvh_to_vrm.frame_rot).
        /// </summary>
        public static bool TryFrameFromRefs(
            Vector3 u1, Vector3 u2, Vector3 w1, Vector3 w2, out Quaternion rotation)
        {
            rotation = Quaternion.identity;
            if (u1.magnitude < Eps || u2.magnitude < Eps || w1.magnitude < Eps || w2.magnitude < Eps)
            {
                return false;
            }
            var fu = FrameFromBasis(u1, u2, Vector3.Cross(u1, u2));
            var fw = FrameFromBasis(w1, w2, Vector3.Cross(w1, w2));
            rotation = fw * Quaternion.Inverse(fu);
            return true;
        }

        /// <summary>Rotation whose basis columns are x, y, z.</summary>
        public static Quaternion FrameFromBasis(Vector3 x, Vector3 y, Vector3 z)
        {
            var matrix = Matrix4x4.identity;
            matrix.SetColumn(0, x);
            matrix.SetColumn(1, y);
            matrix.SetColumn(2, z);
            return matrix.rotation;
        }

        /// <summary>Uniform target/source ratio from the hip->ankle leg span.</summary>
        public static float LegScale(
            Vector3?[] sourceRestPositions,
            Vector3?[] targetRestPositions)
        {
            float Span(Vector3?[] positions, int hip, int ankle)
            {
                if (!positions[hip].HasValue || !positions[ankle].HasValue)
                {
                    return 0f;
                }
                return Vector3.Distance(positions[hip].Value, positions[ankle].Value);
            }

            var source = (Span(sourceRestPositions, MsflowSkeleton.LeftHip, MsflowSkeleton.LeftAnkle) +
                          Span(sourceRestPositions, MsflowSkeleton.RightHip, MsflowSkeleton.RightAnkle)) * 0.5f;
            var target = (Span(targetRestPositions, MsflowSkeleton.LeftHip, MsflowSkeleton.LeftAnkle) +
                          Span(targetRestPositions, MsflowSkeleton.RightHip, MsflowSkeleton.RightAnkle)) * 0.5f;
            if (source < Eps || target < Eps)
            {
                return 1f;
            }
            return target / source;
        }
    }

    /// <summary>
    /// Per-play rest data: bone-local rest directions (u1), twist references
    /// (u2), leaf rest-relative offsets, and the pelvis "back" axis used by the
    /// straight-knee fallback. Built once from the rig's captured rest.
    /// </summary>
    public sealed class RestFrameSet
    {
        public readonly Vector3[] U1 = new Vector3[MsflowSkeleton.JointCount];
        public readonly Vector3[] U2 = new Vector3[MsflowSkeleton.JointCount];
        public readonly bool[] HasU2 = new bool[MsflowSkeleton.JointCount];

        /// <summary>
        /// Constant anatomical twist priors in bone-local space (character
        /// forward for arms, character back for thighs) — used instead of the
        /// structural analog on straight-limb rigs where it degenerates
        /// (bvh_to_vrm's "elbows bend forward / knees bend backward" rules).
        /// </summary>
        public readonly Vector3[] PriorAxisLocal = new Vector3[MsflowSkeleton.JointCount];
        public readonly bool[] HasPrior = new bool[MsflowSkeleton.JointCount];

        public readonly Quaternion[] AuthoredWorld = new Quaternion[MsflowSkeleton.JointCount];
        public readonly Quaternion[] LeafRelToParent = new Quaternion[MsflowSkeleton.JointCount];
        public readonly int[] LeafParent = new int[MsflowSkeleton.JointCount];
        public Vector3 ThighFallbackAxisLocal; // pelvis "back" axis in pelvis-local space
        public bool HasThighFallback;
    }

    /// <summary>Rolling per-pass state (previous world rotations for fallbacks).</summary>
    public sealed class RotationState
    {
        public readonly Quaternion[] PrevWorld = new Quaternion[MsflowSkeleton.JointCount];
        public readonly bool[] HasPrev = new bool[MsflowSkeleton.JointCount];

        public void Reset()
        {
            for (var i = 0; i < MsflowSkeleton.JointCount; i++)
            {
                HasPrev[i] = false;
                PrevWorld[i] = Quaternion.identity;
            }
        }
    }

    /// <summary>
    /// One frame pass: per-joint world rotations (root-local space) from joint
    /// positions, using the anatomical twist references. Leaves inherit their
    /// rest-relative pose inside the nearest mapped ancestor's animated frame.
    /// </summary>
    public static class RotationBuilder
    {
        public static void Build(
            Vector3[] positions,
            int[] successors,
            RestFrameSet rest,
            bool[] mapped,
            RotationState state,
            Quaternion[] worldOut)
        {
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                Quaternion world;
                if (!mapped[joint] || !MsflowRetarget.HasDirectionRef(joint))
                {
                    world = LeafOrUnmapped(joint, rest, state);
                }
                else
                {
                    var tail = successors[joint];
                    var a = positions[joint];
                    var b = tail >= 0 ? positions[tail] : positions[joint];
                    var w1 = b - a;
                    if (w1.magnitude < MsflowRetarget.Eps)
                    {
                        world = TemporalFallback(joint, Vector3.zero, rest, state);
                    }
                    else
                    {
                        w1.Normalize();
                        Vector3 w2;
                        var hasW2 = false;
                        var pair0 = MsflowRetarget.TwistPairs[joint, 0];
                        var pair1 = MsflowRetarget.TwistPairs[joint, 1];
                        if (pair0 >= 0 && pair1 >= 0)
                        {
                            var v2 = positions[pair1] - positions[pair0];
                            hasW2 = MsflowRetarget.TryOrtho(v2, w1, out w2);
                        }
                        else
                        {
                            w2 = Vector3.zero;
                        }
                        if (!hasW2 && (joint == MsflowSkeleton.LeftHip || joint == MsflowSkeleton.RightHip))
                        {
                            // straight knee: the shin carries no bend info, pin
                            // the thigh crease to the pelvis' current backward
                            if (rest.HasThighFallback && state.HasPrev[MsflowSkeleton.Pelvis])
                            {
                                var back = state.PrevWorld[MsflowSkeleton.Pelvis] * rest.ThighFallbackAxisLocal;
                                hasW2 = MsflowRetarget.TryOrtho(back, w1, out w2);
                            }
                        }
                        if (hasW2 && state.HasPrev[joint] && rest.HasU2[joint] &&
                            MsflowRetarget.TwistSmoothing > 0f)
                        {
                            // damp the roll reference toward the previous frame
                            var prevImage = state.PrevWorld[joint] * rest.U2[joint];
                            if (MsflowRetarget.TryOrtho(prevImage, w1, out var prevW2))
                            {
                                w2 = Vector3.Slerp(prevW2, w2, MsflowRetarget.TwistSmoothing).normalized;
                            }
                        }
                        if (hasW2 && rest.HasU2[joint])
                        {
                            if (MsflowRetarget.TryFrameFromRefs(rest.U1[joint], rest.U2[joint], w1, w2, out world))
                            {
                                // ok
                            }
                            else
                            {
                                world = TemporalFallback(joint, w1, rest, state);
                            }
                        }
                        else
                        {
                            world = TemporalFallback(joint, w1, rest, state);
                        }
                    }
                }
                worldOut[joint] = world;
                state.PrevWorld[joint] = world;
                state.HasPrev[joint] = true;
            }
        }

        private static Quaternion LeafOrUnmapped(int joint, RestFrameSet rest, RotationState state)
        {
            var parent = rest.LeafParent[joint];
            if (parent >= 0 && state.HasPrev[parent])
            {
                return state.PrevWorld[parent] * rest.LeafRelToParent[joint];
            }
            if (state.HasPrev[joint])
            {
                return state.PrevWorld[joint];
            }
            return rest.AuthoredWorld[joint];
        }

        private static Quaternion TemporalFallback(
            int joint, Vector3 w1, RestFrameSet rest, RotationState state)
        {
            // keep the previous (or rest) roll, re-aimed onto the direction
            if (w1.magnitude < MsflowRetarget.Eps)
            {
                return state.HasPrev[joint] ? state.PrevWorld[joint] : rest.AuthoredWorld[joint];
            }
            if (!state.HasPrev[joint])
            {
                var restDir = rest.AuthoredWorld[joint] * rest.U1[joint];
                if (restDir.magnitude < MsflowRetarget.Eps)
                {
                    return rest.AuthoredWorld[joint];
                }
                return Quaternion.FromToRotation(restDir, w1) * rest.AuthoredWorld[joint];
            }
            var prev = state.PrevWorld[joint];
            var yPrev = prev * rest.U1[joint];
            if (yPrev.magnitude < MsflowRetarget.Eps)
            {
                return prev;
            }
            return Quaternion.FromToRotation(yPrev, w1) * prev;
        }
    }
}
