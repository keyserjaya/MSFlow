using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace MSFlow.MotionBridge.Tests
{
    /// <summary>
    /// Constraint authoring: the builder must convert Unity world pins into
    /// data-space wire values such that the engine's output mapping brings
    /// them back to the pinned root-local point (exact round-trip through
    /// FromDataConvention), and it must dedupe, validate, and merge like the
    /// request path expects.
    /// </summary>
    public sealed class MsflowConstraintBuilderTests
    {
        private const float ValueTolerance = 1e-5f;
        private const float RoundTripTolerance = 1e-3f;

        [Test]
        public void PinConvertsAndRoundTripsThroughEngine()
        {
            var (root, bones) = BuildCanonicalRig();
            try
            {
                var worldPoint = new Vector3(-0.3f, 0.12f, 0.42f);
                var builder = MsflowConstraintBuilder.ForRig(root.transform, bones);
                builder.PinBone(HumanBodyBones.LeftFoot, 0.5f, worldPoint);
                var controls = builder.Build();

                Assert.AreEqual(1, controls.Count);
                var control = controls[0];
                Assert.AreEqual(0.5f, control.FrameFraction.Value, 1e-6f);
                CollectionAssert.AreEqual(new[] { MsflowSkeleton.LeftAnkle }, control.Joints);

                // data = FromDataConvention(rootWorldToLocal * worldPos) / scale
                Assert.AreEqual(1f, builder.ScaleFactor, 5e-3f, "canonical-size rig, scale 1");
                var rootLocal = root.transform.worldToLocalMatrix.MultiplyPoint3x4(worldPoint);
                var expected = MsflowSkeleton.FromDataConvention(rootLocal) / builder.ScaleFactor;
                Assert.LessOrEqual(
                    Vector3.Distance(expected, RowToVector(control.Values[0])), ValueTolerance,
                    "control value must be FromDataConvention(rootLocal)/scale");

                // engine maps the pinned frame back to the same root-local point
                var frames = DataFramesWithPinnedJoint(
                    3, MsflowSkeleton.LeftAnkle, RowToVector(control.Values[0]));
                var motion = BuildMotionFromData(frames);
                var engine = new MsflowRetargetEngine(
                    motion, root.transform, bones, AllDriven(),
                    alignFacing: false);
                var roundTrip = engine.SourceRootLocalPosition(1f / motion.Fps, MsflowSkeleton.LeftAnkle);
                Assert.LessOrEqual(
                    Vector3.Distance(rootLocal / builder.ScaleFactor, roundTrip), RoundTripTolerance,
                    "engine must map the pinned frame back to the pinned root-local point");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void ScaleNormalizationDividesByLegSpan()
        {
            var scaled = MsflowSkeleton.CanonicalPose();
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                scaled[joint][0] *= 2f;
                scaled[joint][1] *= 2f;
                scaled[joint][2] *= 2f;
            }
            var (root, bones) = BuildRig(scaled);
            try
            {
                var worldPoint = new Vector3(-0.6f, 0.24f, 0.84f);
                var builder = MsflowConstraintBuilder.ForRig(root.transform, bones);
                Assert.AreEqual(2f, builder.ScaleFactor, 1e-2f, "2x rig spans twice the canonical 0.88 m");
                builder.PinBone(HumanBodyBones.LeftFoot, 0.5f, worldPoint);

                var control = builder.Build()[0];
                var rootLocal = root.transform.worldToLocalMatrix.MultiplyPoint3x4(worldPoint);
                var expected = MsflowSkeleton.FromDataConvention(rootLocal) / builder.ScaleFactor;
                Assert.LessOrEqual(
                    Vector3.Distance(expected, RowToVector(control.Values[0])), ValueTolerance,
                    "a 2x rig must pin at half its root-local coordinates (canonical units)");

                var frames = DataFramesWithPinnedJoint(
                    3, MsflowSkeleton.LeftAnkle, RowToVector(control.Values[0]));
                var motion = BuildMotionFromData(frames);
                var engine = new MsflowRetargetEngine(
                    motion, root.transform, bones, AllDriven(), alignFacing: false);
                var roundTrip = engine.SourceRootLocalPosition(1f / motion.Fps, MsflowSkeleton.LeftAnkle);
                Assert.LessOrEqual(
                    Vector3.Distance(rootLocal / builder.ScaleFactor, roundTrip), RoundTripTolerance);

                // and the knob to send raw coordinates
                var raw = MsflowConstraintBuilder.ForRig(root.transform, bones);
                raw.NormalizeScale = false;
                raw.PinBone(HumanBodyBones.LeftFoot, 0.5f, worldPoint);
                Assert.AreEqual(1f, raw.ScaleFactor, 1e-6f);
                var rawValue = RowToVector(raw.Build()[0].Values[0]);
                Assert.LessOrEqual(
                    Vector3.Distance(MsflowSkeleton.FromDataConvention(rootLocal), rawValue),
                    ValueTolerance);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void PinPoseEmitsOnePinPerMappedJoint()
        {
            var (root, bones) = BuildCanonicalRig();
            try
            {
                var builder = MsflowConstraintBuilder.ForRig(root.transform, bones);
                builder.PinPose(0f);
                var controls = builder.Build();

                Assert.AreEqual(1, controls.Count, "same fraction + axes collapses to one entry");
                var control = controls[0];
                Assert.AreEqual(0f, control.FrameFraction.Value, 1e-6f);
                Assert.AreEqual("x,y,z", control.Axes);
                Assert.AreEqual(MsflowSkeleton.JointCount, control.Joints.Length);
                for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
                {
                    Assert.AreEqual(joint, control.Joints[joint], "joints are emitted ascending");
                    var expected = MsflowSkeleton.FromDataConvention(
                                       root.transform.worldToLocalMatrix.MultiplyPoint3x4(
                                           bones[joint].position)) / builder.ScaleFactor;
                    Assert.LessOrEqual(
                        Vector3.Distance(expected, RowToVector(control.Values[joint])), ValueTolerance,
                        $"joint {MsflowSkeleton.JointNames[joint]} value");
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void PinPoseSkipsUnmappedJointsWithOneAggregatedWarning()
        {
            var (root, bones) = BuildCanonicalRig();
            try
            {
                UnityEngine.Object.DestroyImmediate(bones[MsflowSkeleton.LeftFoot].gameObject);
                bones[MsflowSkeleton.LeftFoot] = null;

                var builder = MsflowConstraintBuilder.ForRig(root.transform, bones);
                LogAssert.Expect(
                    LogType.Warning,
                    new Regex("PinPose skipped 1 unmapped joint\\(s\\): left_foot"));
                builder.PinPose(0f);
                LogAssert.NoUnexpectedReceived();

                var control = builder.Build()[0];
                Assert.AreEqual(MsflowSkeleton.JointCount - 1, control.Joints.Length);
                CollectionAssert.DoesNotContain(control.Joints, MsflowSkeleton.LeftFoot);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void PinCurrentPoseAtStartAndEndHitsBothClipEnds()
        {
            var (root, bones) = BuildCanonicalRig();
            try
            {
                var builder = MsflowConstraintBuilder.ForRig(root.transform, bones);
                builder.PinCurrentPoseAtStartAndEnd();
                var controls = builder.Build();

                Assert.AreEqual(2, controls.Count, "one entry per pinned fraction");
                var fractions = new List<float>();
                foreach (var control in controls)
                {
                    fractions.Add(control.FrameFraction.Value);
                    Assert.AreEqual(MsflowSkeleton.JointCount, control.Joints.Length);
                }
                CollectionAssert.AreEqual(new[] { 0f, 1f }, fractions);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void LaterPinWinsOnDuplicateJointAndFraction()
        {
            var (root, bones) = BuildCanonicalRig();
            try
            {
                var first = new Vector3(1f, 1f, 1f);
                var second = new Vector3(2f, 2f, 2f);
                var third = new Vector3(3f, 3f, 3f);
                var builder = MsflowConstraintBuilder.ForRig(root.transform, bones);
                builder.PinJoint(MsflowSkeleton.LeftAnkle, 0.5f, first);
                builder.PinJoint(MsflowSkeleton.LeftAnkle, 0.5004f, second, "y");
                builder.PinJoint(MsflowSkeleton.LeftAnkle, 0.25f, third);
                var controls = builder.Build();

                // 0.5 and 0.5004 collide at the 1e-3 key -> the later pin (y-only) wins
                Assert.AreEqual(2, controls.Count);
                MsflowControl half = null;
                MsflowControl quarter = null;
                foreach (var control in controls)
                {
                    if (Mathf.Abs(control.FrameFraction.Value - 0.5f) < 1e-3f)
                    {
                        half = control;
                    }
                    else
                    {
                        quarter = control;
                    }
                }
                Assert.AreEqual("y", half.Axes);
                Assert.AreEqual(0.5004f, half.FrameFraction.Value, 1e-6f);
                Assert.LessOrEqual(
                    Vector3.Distance(
                        MsflowSkeleton.FromDataConvention(second) / builder.ScaleFactor,
                        RowToVector(half.Values[0])),
                    ValueTolerance);
                Assert.LessOrEqual(
                    Vector3.Distance(
                        MsflowSkeleton.FromDataConvention(third) / builder.ScaleFactor,
                        RowToVector(quarter.Values[0])),
                    ValueTolerance);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void PinBoneThroughoutSpacesPinsEveryIntervalFrame()
        {
            var (root, bones) = BuildCanonicalRig();
            try
            {
                var builder = MsflowConstraintBuilder.ForRig(root.transform, bones);
                builder.FrameCount = 21;
                builder.PinBoneThroughout(HumanBodyBones.Hips, 10);
                var controls = builder.Build();

                Assert.AreEqual(3, controls.Count, "frames 0, 10, 20 of a 21-frame clip");
                var fractions = new List<float>();
                var value = RowToVector(controls[0].Values[0]);
                foreach (var control in controls)
                {
                    fractions.Add(control.FrameFraction.Value);
                    CollectionAssert.AreEqual(new[] { MsflowSkeleton.Pelvis }, control.Joints);
                    Assert.AreEqual("x,y,z", control.Axes);
                    Assert.LessOrEqual(
                        Vector3.Distance(value, RowToVector(control.Values[0])), ValueTolerance,
                        "throughout pins hold the position captured at call time");
                }
                CollectionAssert.AreEqual(new[] { 0f, 0.5f, 1f }, fractions);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void ValidationErrors()
        {
            var (root, bones) = BuildCanonicalRig();
            try
            {
                var builder = MsflowConstraintBuilder.ForRig(root.transform, bones);
                var point = Vector3.zero;

                Assert.Throws<ArgumentOutOfRangeException>(
                    () => builder.PinBone(HumanBodyBones.Hips, -0.01f, point));
                Assert.Throws<ArgumentOutOfRangeException>(
                    () => builder.PinBone(HumanBodyBones.Hips, 1.5f, point));
                Assert.Throws<ArgumentOutOfRangeException>(
                    () => builder.PinBone(HumanBodyBones.Hips, float.NaN, point));
                Assert.Throws<ArgumentOutOfRangeException>(
                    () => builder.PinJoint(-1, 0f, point));
                Assert.Throws<ArgumentOutOfRangeException>(
                    () => builder.PinJoint(MsflowSkeleton.JointCount, 0f, point));
                Assert.Throws<ArgumentOutOfRangeException>(
                    () => builder.PinBoneThroughout(HumanBodyBones.Hips, 0));

                var emptyAxes = Assert.Throws<ArgumentException>(
                    () => builder.PinJoint(0, 0f, point, ""));
                StringAssert.Contains("axes", emptyAxes.Message);
                Assert.Throws<ArgumentException>(
                    () => builder.PinJoint(0, 0f, point, "x,w"));

                var finger = Assert.Throws<ArgumentException>(
                    () => builder.PinBone(HumanBodyBones.LeftMiddleProximal, 0f, point));
                StringAssert.Contains("supported bones", finger.Message);
                StringAssert.Contains("Hips", finger.Message);

                UnityEngine.Object.DestroyImmediate(bones[MsflowSkeleton.LeftFoot].gameObject);
                bones[MsflowSkeleton.LeftFoot] = null;
                var missing = Assert.Throws<ArgumentException>(
                    () => builder.PinBone(HumanBodyBones.LeftToes, 0f, point));
                StringAssert.Contains("LeftToes", missing.Message);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        [Test]
        public void BoneToJointReverseMapMatchesTable()
        {
            Assert.IsTrue(MsflowBoneMapper.TryGetJointForBone(HumanBodyBones.Hips, out var pelvis));
            Assert.AreEqual(MsflowSkeleton.Pelvis, pelvis);
            Assert.IsTrue(MsflowBoneMapper.TryGetJointForBone(HumanBodyBones.LeftFoot, out var ankle));
            Assert.AreEqual(MsflowSkeleton.LeftAnkle, ankle);
            Assert.IsTrue(MsflowBoneMapper.TryGetJointForBone(HumanBodyBones.LeftToes, out var toes));
            Assert.AreEqual(MsflowSkeleton.LeftFoot, toes);
            Assert.IsTrue(MsflowBoneMapper.TryGetJointForBone(HumanBodyBones.UpperChest, out var spine3));
            Assert.AreEqual(MsflowSkeleton.Spine3, spine3);
            Assert.IsTrue(MsflowBoneMapper.TryGetJointForBone(HumanBodyBones.RightHand, out var wrist));
            Assert.AreEqual(MsflowSkeleton.RightWrist, wrist);
            Assert.IsFalse(MsflowBoneMapper.TryGetJointForBone(HumanBodyBones.LeftMiddleProximal, out _));
        }

        [Test]
        public void PinStartPoseMergeKeepsCallerPins()
        {
            var (root, bones) = BuildCanonicalRig();
            var requestObject = new GameObject("MsflowTestRequest");
            try
            {
                var request = requestObject.AddComponent<MsflowMotionRequest>();
                request.PinStartPose = true;
                request.StartPoseSnapshot = () => MsflowConstraintBuilder.ForRig(root.transform, bones);

                var caller = new List<MsflowControl>
                {
                    new MsflowControl
                    {
                        FrameFraction = 0f,
                        Joints = new[] { MsflowSkeleton.Pelvis },
                        Axes = "x,z",
                        Values = new[] { new[] { 9f, 9f, 9f } },
                        Role = "caller",
                    },
                    new MsflowControl
                    {
                        Frame = 0,
                        Joints = new[] { MsflowSkeleton.LeftAnkle },
                        Values = new[] { new[] { 8f, 8f, 8f } },
                    },
                    new MsflowControl
                    {
                        FrameFraction = 0.5f,
                        Joints = new[] { MsflowSkeleton.LeftWrist },
                        Values = new[] { new[] { 7f, 7f, 7f } },
                    },
                };

                var merged = request.ComposeControls(caller);
                Assert.AreEqual(4, merged.Count, "1 split snapshot entry + 3 caller entries");

                var snapshotEntry = merged[0];
                Assert.AreEqual(0f, snapshotEntry.FrameFraction.Value, 1e-6f);
                Assert.AreEqual(MsflowSkeleton.JointCount - 2, snapshotEntry.Joints.Length,
                    "caller pins at frame 0 replace the snapshot's pelvis and left ankle");
                CollectionAssert.DoesNotContain(snapshotEntry.Joints, MsflowSkeleton.Pelvis);
                CollectionAssert.DoesNotContain(snapshotEntry.Joints, MsflowSkeleton.LeftAnkle);

                Assert.AreSame(caller[0], merged[1], "caller entries pass through verbatim");
                Assert.AreSame(caller[1], merged[2]);
                Assert.AreSame(caller[2], merged[3]);

                // the surviving snapshot values still convert through the builder
                var probe = MsflowConstraintBuilder.ForRig(root.transform, bones);
                var expectedNeck = MsflowSkeleton.FromDataConvention(
                    root.transform.worldToLocalMatrix.MultiplyPoint3x4(
                        bones[MsflowSkeleton.Neck].position)) / probe.ScaleFactor;
                var neckIndex = Array.IndexOf(snapshotEntry.Joints, MsflowSkeleton.Neck);
                Assert.GreaterOrEqual(neckIndex, 0);
                Assert.LessOrEqual(
                    Vector3.Distance(expectedNeck, RowToVector(snapshotEntry.Values[neckIndex])),
                    ValueTolerance);

                // without PinStartPose the caller list passes through unchanged
                request.PinStartPose = false;
                var passthrough = request.ComposeControls(caller);
                Assert.AreEqual(3, passthrough.Count);
                Assert.AreSame(caller[0], passthrough[0]);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(requestObject);
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        // -- fixtures -------------------------------------------------------

        private static Vector3 RowToVector(float[] row)
        {
            return new Vector3(row[0], row[1], row[2]);
        }

        private static bool[] AllDriven()
        {
            var driven = new bool[MsflowSkeleton.JointCount];
            for (var joint = 0; joint < driven.Length; joint++)
            {
                driven[joint] = true;
            }
            return driven;
        }

        /// <summary>
        /// Canonical frames in the *data* convention (X-mirrored vs Unity)
        /// with one joint replaced by an already-converted wire value.
        /// </summary>
        private static float[][][] DataFramesWithPinnedJoint(
            int frameCount, int pinnedJoint, Vector3 pinnedValue)
        {
            var canonical = MsflowSkeleton.CanonicalPose();
            var frames = new float[frameCount][][];
            for (var frame = 0; frame < frameCount; frame++)
            {
                frames[frame] = new float[MsflowSkeleton.JointCount][];
                for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
                {
                    frames[frame][joint] = new[]
                    {
                        -canonical[joint][0], canonical[joint][1], canonical[joint][2],
                    };
                }
            }
            var middle = frameCount / 2;
            frames[middle][pinnedJoint] = new[]
            {
                pinnedValue.x, pinnedValue.y, pinnedValue.z,
            };
            return frames;
        }

        /// <summary>22 root transforms at the canonical pose (identity rotations).</summary>
        private static (GameObject root, Transform[] bones) BuildCanonicalRig()
        {
            return BuildRig(MsflowSkeleton.CanonicalPose());
        }

        private static (GameObject root, Transform[] bones) BuildRig(float[][] positions)
        {
            var root = new GameObject("MsflowConstraintTestRoot");
            var bones = new Transform[MsflowSkeleton.JointCount];
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                bones[joint] = new GameObject(
                    $"bone_{joint:00}_{MsflowSkeleton.JointNames[joint]}").transform;
            }
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                var parent = MsflowSkeleton.Parents[joint];
                bones[joint].SetParent(parent < 0 ? root.transform : bones[parent], false);
                bones[joint].position = new Vector3(
                    positions[joint][0], positions[joint][1], positions[joint][2]);
            }
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                if (MsflowSkeleton.Parents[joint] >= 0)
                {
                    bones[joint].localRotation = Quaternion.identity;
                }
            }
            return (root, bones);
        }

        /// <summary>Wire motion from frames already in the data convention.</summary>
        private static MsflowMotion BuildMotionFromData(float[][][] dataFrames)
        {
            var document = new JObject
            {
                ["format"] = "msflow-motion",
                ["version"] = 1,
                ["fps"] = 20,
                ["n_frames"] = dataFrames.Length,
                ["joints"] = new JArray(MsflowSkeleton.JointNames),
                ["parents"] = new JArray(MsflowSkeleton.Parents),
                ["positions"] = ToJArray(dataFrames),
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
                    jointsArray.Add(new JArray(joint[0], joint[1], joint[2]));
                }
                framesArray.Add(jointsArray);
            }
            return framesArray;
        }
    }
}
