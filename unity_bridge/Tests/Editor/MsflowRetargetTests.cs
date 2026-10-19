using System;
using System.Collections.Generic;
using System.Text;
using NUnit.Framework;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace MSFlow.MotionBridge.Tests
{
    /// <summary>
    /// Core correctness check for the C# port of retarget.py: a synthetic rig
    /// whose rest pose IS the canonical source skeleton must round-trip the
    /// source motion — applying the evaluated pose must put every joint's
    /// world position back onto the source joint positions (same property the
    /// Blender retarget selftest asserts, ported to Unity).
    /// </summary>
    public sealed class MsflowRetargetTests
    {
        private const float Tolerance = 5e-3f;

        [Test]
        public void RestFrameKeepsAuthoredPose()
        {
            var (root, bones) = BuildCanonicalRig();
            try
            {
                var motion = BuildMotion(new[] { CanonicalPositions(), CanonicalPositions() });
                var engine = new MsflowRetargetEngine(
                    motion, root.transform, bones, AllDriven());
                var pose = engine.Evaluate(0f);
                ApplyPose(root.transform, engine, pose, 1f);

                var canonical = CanonicalPositions();
                for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
                {
                    var expected = new Vector3(
                        canonical[joint][0], canonical[joint][1], canonical[joint][2]);
                    Assert.LessOrEqual(
                        Vector3.Distance(expected, bones[joint].position), Tolerance,
                        $"joint {joint} {MsflowSkeleton.JointNames[joint]} drifted at rest");
                    Assert.LessOrEqual(
                        Quaternion.Angle(bones[joint].localRotation, Quaternion.identity), 0.5f,
                        $"joint {joint} should keep its authored rest rotation");
                }
                Assert.AreEqual(1f, engine.ScaleFactor, 1e-4f);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void RotatedArmRoundTripsToSourcePositions()
        {
            var (root, bones) = BuildCanonicalRig();
            try
            {
                var reference = CanonicalPositions();
                var rotated = RotateLeftArmZ(reference, 30f);
                var motion = BuildMotion(new[] { reference, rotated, rotated });
                var engine = new MsflowRetargetEngine(
                    motion, root.transform, bones, AllDriven());

                // at the reference frame the pose is the authored rest
                var restPose = engine.Evaluate(0f);
                ApplyPose(root.transform, engine, restPose, 1f);
                for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
                {
                    Assert.LessOrEqual(
                        Quaternion.Angle(bones[joint].localRotation, Quaternion.identity), 0.5f,
                        $"rest frame must not rotate joint {joint}");
                }

                // one frame in: FK head positions must land on the source joints
                var pose = engine.Evaluate(1f / motion.Fps);
                ApplyPose(root.transform, engine, pose, 1f);
                for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
                {
                    var expected = new Vector3(
                        rotated[joint][0], rotated[joint][1], rotated[joint][2]);
                    var distance = Vector3.Distance(expected, bones[joint].position);
                    Assert.LessOrEqual(
                        distance, Tolerance,
                        $"joint {joint} {MsflowSkeleton.JointNames[joint]} off by {distance}");
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void LegScaleComparesSourceAndTargetSpans()
        {
            var source = CanonicalPositions();
            var compressed = CanonicalPositions();
            foreach (var joint in new[]
                     {
                         MsflowSkeleton.LeftHip, MsflowSkeleton.RightHip,
                         MsflowSkeleton.LeftKnee, MsflowSkeleton.RightKnee,
                         MsflowSkeleton.LeftAnkle, MsflowSkeleton.RightAnkle,
                         MsflowSkeleton.LeftFoot, MsflowSkeleton.RightFoot,
                     })
            {
                compressed[joint][0] *= 0.8f;
                compressed[joint][1] *= 0.8f;
                compressed[joint][2] *= 0.8f;
            }
            var sourceArray = ToNullable(source);
            var targetArray = ToNullable(compressed);
            var scale = MsflowRetarget.LegScale(targetArray, sourceArray);
            Assert.AreEqual(1f / 0.8f, scale, 1e-3f);

            // degenerate input falls back to 1
            var empty = new Vector3?[MsflowSkeleton.JointCount];
            Assert.AreEqual(1f, MsflowRetarget.LegScale(empty, empty), 1e-6f);
        }

        [Test]
        public void FrameFromRefsAimsAndTwists()
        {
            var u1 = new Vector3(0f, 1f, 0f);
            var u2 = new Vector3(1f, 0f, 0f);
            var w1 = new Vector3(0f, 0f, 1f);
            var w2 = new Vector3(0f, 1f, 0f);
            var ok = MsflowRetarget.TryFrameFromRefs(u1, u2, w1, w2, out var rotation);
            Assert.IsTrue(ok);
            Assert.LessOrEqual(Vector3.Angle(rotation * u1, w1), 1e-3f, "u1 must map to w1");
            Assert.LessOrEqual(Vector3.Angle(rotation * u2, w2), 1e-3f, "u2 must map to w2");
            Assert.IsFalse(
                MsflowRetarget.TryFrameFromRefs(Vector3.zero, u2, w1, w2, out _),
                "degenerate inputs must be rejected");
        }

        [Test]
        public void TwistFollowsShoulderLineOnTPoseRig()
        {
            // T-pose rest + a source whose upper body twists about the spine
            // while the spine direction stays fixed: the posed shoulder
            // positions must follow the source laterally (roll must track the
            // collar-line twist reference, not freeze at the rest roll).
            var (root, bones) = BuildRig(TPosePositions());
            try
            {
                var reference = CanonicalPositions();
                var twisted = (float[][])reference.Clone();
                for (var i = 0; i < reference.Length; i++)
                {
                    twisted[i] = (float[])reference[i].Clone();
                }
                var twist = Quaternion.Euler(0f, 60f, 0f);
                var pivot = new Vector3(
                    reference[MsflowSkeleton.Pelvis][0],
                    reference[MsflowSkeleton.Pelvis][1],
                    reference[MsflowSkeleton.Pelvis][2]);
                foreach (var joint in new[]
                         {
                             MsflowSkeleton.Neck, MsflowSkeleton.LeftCollar, MsflowSkeleton.RightCollar,
                             MsflowSkeleton.Head, MsflowSkeleton.LeftShoulder, MsflowSkeleton.RightShoulder,
                             MsflowSkeleton.LeftElbow, MsflowSkeleton.RightElbow,
                             MsflowSkeleton.LeftWrist, MsflowSkeleton.RightWrist,
                         })
                {
                    var p = new Vector3(
                        reference[joint][0], reference[joint][1], reference[joint][2]);
                    var moved = pivot + twist * (p - pivot);
                    twisted[joint][0] = moved.x;
                    twisted[joint][1] = moved.y;
                    twisted[joint][2] = moved.z;
                }
                var motion = BuildMotion(new[] { reference, twisted, twisted });
                var engine = new MsflowRetargetEngine(
                    motion, root.transform, bones, AllDriven());
                var pose = engine.Evaluate(1f / motion.Fps);
                ApplyPose(root.transform, engine, pose, 1f);
                for (var i = 0; i < pose.Bones.Count; i++)
                {
                    var joint = pose.Bones[i].Joint;
                    var expected = new Vector3(
                        twisted[joint][0], twisted[joint][1], twisted[joint][2]);
                    var distance = Vector3.Distance(expected, bones[joint].position);
                    Assert.LessOrEqual(
                        distance, Tolerance,
                        $"joint {joint} {MsflowSkeleton.JointNames[joint]} off by {distance} " +
                        "(twist reference must follow the shoulder line)");
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void IdentityMotionLeavesRigStill()
        {
            var (root, bones) = BuildCanonicalRig();
            try
            {
                // arbitrary non-identity authored rotations — an identity-rest
                // rig cannot exercise conjugation bugs in the rest alignment
                for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
                {
                    bones[joint].localRotation = Quaternion.Euler(7f * joint, 11f * joint, 5f * joint);
                }
                var captured = new float[MsflowSkeleton.JointCount][];
                var authored = new Quaternion[MsflowSkeleton.JointCount];
                for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
                {
                    var point = bones[joint].position;
                    captured[joint] = new[] { point.x, point.y, point.z };
                    authored[joint] = bones[joint].rotation;
                }
                // a motion whose frames are exactly the captured pose must not
                // move the rig at any weight
                var motion = BuildMotion(new[] { captured, captured });
                var engine = new MsflowRetargetEngine(
                    motion, root.transform, bones, AllDriven());
                ApplyPose(root.transform, engine, engine.Evaluate(0f), 1f);
                for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
                {
                    Assert.LessOrEqual(
                        Quaternion.Angle(bones[joint].rotation, authored[joint]), 0.5f,
                        $"joint {joint} moved although the motion equals its captured pose");
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void TPoseRestRigTracksSourceDirections()
        {
            var (root, bones) = BuildRig(TPosePositions());
            try
            {
                var reference = CanonicalPositions();
                var raised = RotateLeftArmZ(reference, -45f);
                var motion = BuildMotion(new[] { reference, raised, raised });
                var engine = new MsflowRetargetEngine(
                    motion, root.transform, bones, AllDriven());

                ApplyPose(root.transform, engine, engine.Evaluate(0f), 1f);
                Assert.GreaterOrEqual(
                    WorstDirectionDot(bones, reference), 0.98f,
                    "bone directions must track the source at the ref frame");

                ApplyPose(root.transform, engine, engine.Evaluate(1f / motion.Fps), 1f);
                Assert.GreaterOrEqual(
                    WorstDirectionDot(bones, raised), 0.98f,
                    "bone directions must track the source after the arm raise");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void RestAnchoredKeepsRigRestAtRefFrame()
        {
            var (root, bones) = BuildRig(TPosePositions());
            try
            {
                var authored = new Quaternion[MsflowSkeleton.JointCount];
                for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
                {
                    authored[joint] = bones[joint].rotation;
                }
                var reference = CanonicalPositions();
                var raised = RotateLeftArmZ(reference, -45f);
                var motion = BuildMotion(new[] { reference, raised, raised });
                var engine = new MsflowRetargetEngine(
                    motion, root.transform, bones, AllDriven(),
                    MsflowRootMotionMode.Hips, true, MsflowPoseTransferMode.RestAnchored);
                ApplyPose(root.transform, engine, engine.Evaluate(0f), 1f);
                for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
                {
                    Assert.LessOrEqual(
                        Quaternion.Angle(bones[joint].rotation, authored[joint]), 0.5f,
                        $"RestAnchored must keep joint {joint} at its captured rest at the ref frame");
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void ElbowBendRollFollowsBendPlane()
        {
            // same forearm aim, two different bend planes: the hand's roll must
            // follow the bend (~90 deg apart), not freeze at the rest roll
            // (the straight-arm rest makes the structural twist analog
            // degenerate — the constant prior must carry the roll).
            var previous = MsflowRetarget.TwistSmoothing;
            MsflowRetarget.TwistSmoothing = 0f;
            var (root, bones) = BuildCanonicalRig();
            try
            {
                var downBend = CanonicalPositions(); // elbow below shoulder, forearm forward
                downBend[MsflowSkeleton.LeftWrist] = new[] { -0.20f, 1.12f, 0.24f };
                var outBend = CanonicalPositions();  // elbow out to the side, forearm forward
                outBend[MsflowSkeleton.LeftElbow] = new[] { -0.48f, 1.40f, 0.02f };
                outBend[MsflowSkeleton.LeftWrist] = new[] { -0.48f, 1.40f, 0.24f };

                var motionA = BuildMotion(new[] { downBend, downBend, downBend });
                var engineA = new MsflowRetargetEngine(
                    motionA, root.transform, bones, AllDriven());
                ApplyPose(root.transform, engineA, engineA.Evaluate(1f / motionA.Fps), 1f);
                var handA = bones[MsflowSkeleton.LeftWrist].rotation;
                UnityEngine.Object.DestroyImmediate(root);

                (root, bones) = BuildCanonicalRig();
                var motionB = BuildMotion(new[] { outBend, outBend, outBend });
                var engineB = new MsflowRetargetEngine(
                    motionB, root.transform, bones, AllDriven());
                ApplyPose(root.transform, engineB, engineB.Evaluate(1f / motionB.Fps), 1f);
                var handB = bones[MsflowSkeleton.LeftWrist].rotation;

                var angle = Quaternion.Angle(handA, handB);
                Assert.That(angle, Is.InRange(60f, 120f),
                    $"hand roll must follow the bend plane (got {angle:0.0} deg)");
            }
            finally
            {
                MsflowRetarget.TwistSmoothing = previous;
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void TwistSmoothingDampsRollJitter()
        {
            var previous = MsflowRetarget.TwistSmoothing;
            var (root, bones) = BuildCanonicalRig();
            try
            {
                var downBend = CanonicalPositions();
                downBend[MsflowSkeleton.LeftWrist] = new[] { -0.20f, 1.12f, 0.24f };
                var outBend = CanonicalPositions();
                outBend[MsflowSkeleton.LeftElbow] = new[] { -0.48f, 1.40f, 0.02f };
                outBend[MsflowSkeleton.LeftWrist] = new[] { -0.48f, 1.40f, 0.24f };
                var motion = BuildMotion(new[] { downBend, outBend, outBend });

                MsflowRetarget.TwistSmoothing = 0f;
                var raw = MeasureHandRollDelta(root.transform, bones, motion);
                UnityEngine.Object.DestroyImmediate(root);

                (root, bones) = BuildCanonicalRig();
                MsflowRetarget.TwistSmoothing = 0.5f;
                var smoothed = MeasureHandRollDelta(root.transform, bones, motion);

                Assert.Less(smoothed, raw * 0.85f,
                    $"smoothing must damp frame-to-frame roll (raw {raw:0.0} -> smoothed {smoothed:0.0})");
            }
            finally
            {
                MsflowRetarget.TwistSmoothing = previous;
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static float MeasureHandRollDelta(
            Transform root, Transform[] bones, MsflowMotion motion)
        {
            var engine = new MsflowRetargetEngine(motion, root, bones, AllDriven());
            ApplyPose(root, engine, engine.Evaluate(0f), 1f);
            var first = bones[MsflowSkeleton.LeftWrist].rotation;
            ApplyPose(root, engine, engine.Evaluate(1f / motion.Fps), 1f);
            var second = bones[MsflowSkeleton.LeftWrist].rotation;
            return Quaternion.Angle(first, second);
        }

        [Test]
        public void FacingAlignmentAnchorsStartFacing()
        {
            // generated clips can face ~180 deg from the rig: with AlignFacing
            // the source's start facing is yawed onto the rig's rest facing
            // (hips must not carry the 180 deg), and internal turns survive.
            var (root, bones) = BuildCanonicalRig();
            try
            {
                var backward = RotateAllYaw(CanonicalPositions(), 180f);
                var motion = BuildMotion(new[] { backward, backward, backward });

                var aligned = new MsflowRetargetEngine(
                    motion, root.transform, bones, AllDriven(),
                    MsflowRootMotionMode.Hips, true,
                    MsflowPoseTransferMode.MatchSourcePose, true);
                ApplyPose(root.transform, aligned, aligned.Evaluate(0f), 1f);
                Assert.LessOrEqual(
                    Quaternion.Angle(bones[MsflowSkeleton.Pelvis].rotation, Quaternion.identity), 2f,
                    "aligned playback must not carry the source's 180 deg facing");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }

            (root, bones) = BuildCanonicalRig();
            try
            {
                var backward = RotateAllYaw(CanonicalPositions(), 180f);
                var motion = BuildMotion(new[] { backward, backward, backward });
                var raw = new MsflowRetargetEngine(
                    motion, root.transform, bones, AllDriven(),
                    MsflowRootMotionMode.Hips, true,
                    MsflowPoseTransferMode.MatchSourcePose, false);
                ApplyPose(root.transform, raw, raw.Evaluate(0f), 1f);
                Assert.GreaterOrEqual(
                    Quaternion.Angle(bones[MsflowSkeleton.Pelvis].rotation, Quaternion.identity), 170f,
                    "without AlignFacing the raw source facing (180 deg) must show");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void SpineSmoothingFlattensZigzagKeepsLean()
        {
            // source with a crunchy spine: segment 2 leans back, segment 3
            // forward (the V-shape seen in generated clips)
            var zigzag = CanonicalPositions();
            zigzag[MsflowSkeleton.Spine3] = new[] { 0f, 1.20f, -0.10f };   // 6>9 leans back
            zigzag[MsflowSkeleton.Neck] = new[] { 0f, 1.42f, 0.05f };      // 9>12 leans forward
            var motion = BuildMotion(new[] { zigzag, zigzag, zigzag });

            var (root, bones) = BuildCanonicalRig();
            try
            {
                var engine = new MsflowRetargetEngine(
                    motion, root.transform, bones, AllDriven(),
                    MsflowRootMotionMode.Hips, true,
                    MsflowPoseTransferMode.MatchSourcePose, true, 1f);
                ApplyPose(root.transform, engine, engine.Evaluate(1f / motion.Fps), 1f);

                var d1 = bones[MsflowSkeleton.Spine2].position - bones[MsflowSkeleton.Spine1].position;
                var d2 = bones[MsflowSkeleton.Spine3].position - bones[MsflowSkeleton.Spine2].position;
                var d3 = bones[MsflowSkeleton.Neck].position - bones[MsflowSkeleton.Spine3].position;
                var bend12 = Vector3.Angle(d1, d2);
                var bend23 = Vector3.Angle(d2, d3);
                Assert.Less(Mathf.Abs(bend12 - bend23), 10f,
                    $"relative bends must equalize (got {bend12:0.0} / {bend23:0.0})");
                var totalPosed = Quaternion.FromToRotation(d1, d3);
                var s1 = new Vector3(0f, 0.08f, 0f);
                var s3 = zigzag[MsflowSkeleton.Neck][2] - zigzag[MsflowSkeleton.Spine3][2];
                var totalSource = Quaternion.FromToRotation(s1, new Vector3(0f, 0.22f, s3));
                Assert.Less(Quaternion.Angle(totalPosed, totalSource), 5f,
                    "the overall torso lean must be preserved");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        // -- helpers ---------------------------------------------------------

        private static bool[] AllDriven()
        {
            var driven = new bool[MsflowSkeleton.JointCount];
            for (var i = 0; i < driven.Length; i++)
            {
                driven[i] = true;
            }
            return driven;
        }

        private static float[][] CanonicalPositions()
        {
            return MsflowSkeleton.CanonicalPose();
        }

        private static float[][] RotateAllYaw(float[][] source, float degrees)
        {
            var rotated = new float[MsflowSkeleton.JointCount][];
            var rotation = Quaternion.Euler(0f, degrees, 0f);
            for (var i = 0; i < source.Length; i++)
            {
                var p = new Vector3(source[i][0], source[i][1], source[i][2]);
                var moved = rotation * p;
                rotated[i] = new[] { moved.x, moved.y, moved.z };
            }
            return rotated;
        }

        private static float[][] RotateLeftArmZ(float[][] source, float degrees)
        {
            return RotateChainZ(source, degrees,
                MsflowSkeleton.LeftShoulder, MsflowSkeleton.LeftElbow, MsflowSkeleton.LeftWrist);
        }

        private static float[][] RotateRightArmZ(float[][] source, float degrees)
        {
            return RotateChainZ(source, degrees,
                MsflowSkeleton.RightShoulder, MsflowSkeleton.RightElbow, MsflowSkeleton.RightWrist);
        }

        /// <summary>Canonical rest with both arms swung out to a T-Pose.</summary>
        private static float[][] TPosePositions()
        {
            var pose = RotateLeftArmZ(CanonicalPositions(), -90f);
            return RotateRightArmZ(pose, 90f);
        }

        private static float[][] RotateChainZ(
            float[][] source, float degrees, int pivotJoint, params int[] movingJoints)
        {
            var rotated = new float[MsflowSkeleton.JointCount][];
            for (var i = 0; i < source.Length; i++)
            {
                rotated[i] = (float[])source[i].Clone();
            }
            var pivot = new Vector3(
                source[pivotJoint][0], source[pivotJoint][1], source[pivotJoint][2]);
            var rotation = Quaternion.Euler(0f, 0f, degrees);
            foreach (var joint in movingJoints)
            {
                var position = new Vector3(
                    source[joint][0], source[joint][1], source[joint][2]);
                var moved = pivot + rotation * (position - pivot);
                rotated[joint][0] = moved.x;
                rotated[joint][1] = moved.y;
                rotated[joint][2] = moved.z;
            }
            return rotated;
        }

        /// <summary>Worst cosine between each driven bone direction and the source segment.</summary>
        private static float WorstDirectionDot(Transform[] bones, float[][] expected)
        {
            var successors = MsflowSkeleton.ChainSuccessors();
            var worst = 1f;
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                var tail = successors[joint];
                if (tail < 0)
                {
                    continue;
                }
                var boneDir = bones[tail].position - bones[joint].position;
                var sourceDir = new Vector3(
                    expected[tail][0] - expected[joint][0],
                    expected[tail][1] - expected[joint][1],
                    expected[tail][2] - expected[joint][2]);
                if (boneDir.magnitude < 1e-6f || sourceDir.magnitude < 1e-6f)
                {
                    continue;
                }
                worst = Mathf.Min(worst, Vector3.Dot(boneDir.normalized, sourceDir.normalized));
            }
            return worst;
        }

        private static Vector3?[] ToNullable(float[][] positions)
        {
            var array = new Vector3?[positions.Length];
            for (var i = 0; i < positions.Length; i++)
            {
                array[i] = new Vector3(positions[i][0], positions[i][1], positions[i][2]);
            }
            return array;
        }

        /// <summary>
        /// 22 root transforms at the canonical pose (parents per the source
        /// tree, identity rotations) — a stand-in rig that matches the source.
        /// </summary>
        private static (GameObject root, Transform[] bones) BuildCanonicalRig()
        {
            return BuildRig(CanonicalPositions());
        }

        /// <summary>Stand-in rig whose rest bone heads sit at <paramref name="positions"/>.</summary>
        private static (GameObject root, Transform[] bones) BuildRig(float[][] positions)
        {
            var root = new GameObject("MsflowTestRoot");
            var bones = new Transform[MsflowSkeleton.JointCount];
            var transforms = new Transform[MsflowSkeleton.JointCount];
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                var go = new GameObject($"bone_{joint:00}_{MsflowSkeleton.JointNames[joint]}");
                transforms[joint] = go.transform;
            }
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                var parent = MsflowSkeleton.Parents[joint];
                var holder = parent < 0 ? root.transform : transforms[parent];
                transforms[joint].SetParent(holder, false);
                bones[joint] = transforms[joint];
                var world = new Vector3(
                    positions[joint][0], positions[joint][1], positions[joint][2]);
                transforms[joint].position = world;
            }
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                var parent = MsflowSkeleton.Parents[joint];
                if (parent >= 0)
                {
                    transforms[joint].localRotation = Quaternion.identity;
                }
            }
            return (root, bones);
        }

        /// <summary>
        /// Driver-equivalent application (weight 1): compose desired world
        /// rotations into local rotations down the hierarchy, FK order.
        /// </summary>
        private static void ApplyPose(
            Transform root, MsflowRetargetEngine engine, MsflowPose pose, float weight)
        {
            foreach (var target in pose.Bones)
            {
                var bone = target.Bone;
                var worldTarget = engine.RootRotation * target.WorldRotationRootLocal;
                var parent = bone.parent;
                var parentWorld = parent != null ? parent.rotation : Quaternion.identity;
                var local = Quaternion.Inverse(parentWorld) * worldTarget;
                bone.localRotation = Quaternion.Slerp(bone.localRotation, local, weight);
            }
            var hips = root;
            for (var i = 0; i < pose.Bones.Count; i++)
            {
                if (pose.Bones[i].Joint == MsflowSkeleton.Pelvis)
                {
                    hips = pose.Bones[i].Bone;
                    break;
                }
            }
            if (hips != null && hips.parent != null)
            {
                var worldTarget = engine.HipsWorldTarget(pose.PelvisDeltaRootLocal);
                hips.localPosition = Vector3.Lerp(
                    hips.localPosition,
                    hips.parent.InverseTransformPoint(worldTarget), weight);
            }
        }

        private static MsflowMotion BuildMotion(float[][][] frames)
        {
            var document = new JObject
            {
                ["format"] = "msflow-motion",
                ["version"] = 1,
                ["fps"] = 20,
                ["n_frames"] = frames.Length,
                ["joints"] = new JArray(MsflowSkeleton.JointNames),
                ["parents"] = new JArray(MsflowSkeleton.Parents),
                ["positions"] = ToJArray(frames),
            };
            return MsflowMotion.FromJson(document.ToString(Formatting.None));
        }

        private static JArray ToJArray(float[][][] frames)
        {
            var framesArray = new JArray();
            foreach (var frame in frames)
            {
                var jointsArray = new JArray();
                foreach (var joint in frame)
                {
                    // serialize in the data's convention (X-mirrored vs Unity,
                    // see MsflowSkeleton.FromDataConvention): the engine
                    // converts back, so assertions stay in Unity semantics.
                    jointsArray.Add(new JArray(-joint[0], joint[1], joint[2]));
                }
                framesArray.Add(jointsArray);
            }
            return framesArray;
        }
    }
}
