using System;
using NUnit.Framework;
using UnityEngine;

namespace MSFlow.MotionBridge.Tests
{
    public sealed class MsflowMappingTests
    {
        [Test]
        public void SpineChainSpreadsAcrossAvailableSlots()
        {
            var spine = new GameObject("Spine").transform;
            var chest = new GameObject("Chest").transform;
            var upper = new GameObject("UpperChest").transform;
            try
            {
                var full = MsflowBoneMapper.SelectSpineChain(spine, chest, upper);
                Assert.AreEqual(spine, full[0]);
                Assert.AreEqual(chest, full[1]);
                Assert.AreEqual(upper, full[2]);

                var two = MsflowBoneMapper.SelectSpineChain(spine, chest, null);
                Assert.AreEqual(spine, two[0]);
                Assert.IsNull(two[1]);
                Assert.AreEqual(chest, two[2]);

                var one = MsflowBoneMapper.SelectSpineChain(spine, null, null);
                Assert.AreEqual(spine, one[0]);
                Assert.IsNull(one[1]);
                Assert.IsNull(one[2]);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(spine.gameObject);
                UnityEngine.Object.DestroyImmediate(chest.gameObject);
                UnityEngine.Object.DestroyImmediate(upper.gameObject);
            }
        }

        [Test]
        public void MissingRequiredBonesAreListed()
        {
            var mapping = new MsflowBoneMapping();
            mapping.Bones[MsflowSkeleton.Pelvis] = new GameObject("Hips").transform;
            // everything else missing, including all spine slots
            var exc = Assert.Throws<InvalidOperationException>(
                () => MsflowBoneMapper.ValidateRequired(mapping));
            StringAssert.Contains("head", exc.Message);
            StringAssert.Contains("spine1/spine2/spine3", exc.Message);
        }

        [Test]
        public void CompleteManualMappingValidates()
        {
            var mapping = new MsflowBoneMapping();
            var transforms = new Transform[MsflowSkeleton.JointCount];
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                transforms[joint] = new GameObject($"bone_{joint:00}").transform;
                mapping.Bones[joint] = transforms[joint];
            }
            Assert.DoesNotThrow(() => MsflowBoneMapper.ValidateRequired(mapping));
            cleanup(transforms);
        }

        [Test]
        public void AllMappedBonesAreDriven()
        {
            var mapping = new MsflowBoneMapping();
            var transforms = new Transform[MsflowSkeleton.JointCount];
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                transforms[joint] = new GameObject($"bone_{joint:00}").transform;
                mapping.Bones[joint] = transforms[joint];
            }
            MsflowBoneMapper.ApplyDrivenPolicy(mapping);
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                Assert.IsTrue(mapping.Driven[joint],
                    $"{MsflowSkeleton.JointNames[joint]} must be driven — " +
                    "an undriven parent leaves its children at the Animator's angles");
            }
            // the joints an earlier policy wrongly left as geometry-only
            Assert.IsTrue(mapping.Driven[MsflowSkeleton.Neck]);
            Assert.IsTrue(mapping.Driven[MsflowSkeleton.LeftFoot]);
            Assert.IsTrue(mapping.Driven[MsflowSkeleton.RightFoot]);

            // joints without a bone are never driven
            UnityEngine.Object.DestroyImmediate(transforms[MsflowSkeleton.Neck].gameObject);
            mapping.Bones[MsflowSkeleton.Neck] = null;
            MsflowBoneMapper.ApplyDrivenPolicy(mapping);
            Assert.IsFalse(mapping.Driven[MsflowSkeleton.Neck]);
            cleanup(transforms);
        }

        private static void cleanup(Transform[] transforms)
        {
            foreach (var transform in transforms)
            {
                if (transform != null)
                {
                    UnityEngine.Object.DestroyImmediate(transform.gameObject);
                }
            }
        }
    }
}
