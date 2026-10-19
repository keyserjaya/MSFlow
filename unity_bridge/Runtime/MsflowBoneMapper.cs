using System;
using System.Collections.Generic;
using UnityEngine;

namespace MSFlow.MotionBridge
{
    /// <summary>
    /// Resolved 22-joint mapping for one character. <see cref="Bones"/> holds
    /// every humanoid bone that exists on the avatar (geometry, used for
    /// reference frames); <see cref="Driven"/> marks which of those the
    /// playback driver rotates.
    /// </summary>
    public sealed class MsflowBoneMapping
    {
        public readonly Transform[] Bones = new Transform[MsflowSkeleton.JointCount];
        public readonly bool[] Driven = new bool[MsflowSkeleton.JointCount];
        public readonly List<string> Warnings = new List<string>();
    }

    /// <summary>
    /// Maps MSFlow's 22 source joints onto a Unity Humanoid avatar.
    /// Required bones (missing = play refused): hips, head, both upper/lower
    /// legs + feet, both upper/lower arms + hands, and at least one spine
    /// slot. Optional: chest/upper chest (spine-chain spread), shoulders,
    /// neck and toes. Every mapped bone is driven — including neck and toes:
    /// an undriven bone would leave its children's positions at the
    /// Animator's angles (e.g. the neck rotation positions the head bone).
    /// </summary>
    public static class MsflowBoneMapper
    {
        private static readonly int[] RequiredJoints =
        {
            MsflowSkeleton.Pelvis,
            MsflowSkeleton.Head,
            MsflowSkeleton.LeftHip, MsflowSkeleton.RightHip,
            MsflowSkeleton.LeftKnee, MsflowSkeleton.RightKnee,
            MsflowSkeleton.LeftAnkle, MsflowSkeleton.RightAnkle,
            MsflowSkeleton.LeftShoulder, MsflowSkeleton.RightShoulder,
            MsflowSkeleton.LeftElbow, MsflowSkeleton.RightElbow,
            MsflowSkeleton.LeftWrist, MsflowSkeleton.RightWrist,
        };

        /// <summary>Humanoid bones with an MSFlow source joint (the table above, inverted).</summary>
        public static readonly HumanBodyBones[] SupportedBones =
        {
            HumanBodyBones.Hips,
            HumanBodyBones.LeftUpperLeg, HumanBodyBones.RightUpperLeg,
            HumanBodyBones.Spine, HumanBodyBones.LeftLowerLeg, HumanBodyBones.RightLowerLeg,
            HumanBodyBones.Chest, HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot,
            HumanBodyBones.UpperChest, HumanBodyBones.LeftToes, HumanBodyBones.RightToes,
            HumanBodyBones.Neck, HumanBodyBones.LeftShoulder, HumanBodyBones.RightShoulder,
            HumanBodyBones.Head, HumanBodyBones.LeftUpperArm, HumanBodyBones.RightUpperArm,
            HumanBodyBones.LeftLowerArm, HumanBodyBones.RightLowerArm,
            HumanBodyBones.LeftHand, HumanBodyBones.RightHand,
        };

        private static readonly Dictionary<HumanBodyBones, int> JointForBone =
            new Dictionary<HumanBodyBones, int>
            {
                { HumanBodyBones.Hips, MsflowSkeleton.Pelvis },
                { HumanBodyBones.LeftUpperLeg, MsflowSkeleton.LeftHip },
                { HumanBodyBones.RightUpperLeg, MsflowSkeleton.RightHip },
                { HumanBodyBones.Spine, MsflowSkeleton.Spine1 },
                { HumanBodyBones.LeftLowerLeg, MsflowSkeleton.LeftKnee },
                { HumanBodyBones.RightLowerLeg, MsflowSkeleton.RightKnee },
                { HumanBodyBones.Chest, MsflowSkeleton.Spine2 },
                { HumanBodyBones.LeftFoot, MsflowSkeleton.LeftAnkle },
                { HumanBodyBones.RightFoot, MsflowSkeleton.RightAnkle },
                { HumanBodyBones.UpperChest, MsflowSkeleton.Spine3 },
                { HumanBodyBones.LeftToes, MsflowSkeleton.LeftFoot },
                { HumanBodyBones.RightToes, MsflowSkeleton.RightFoot },
                { HumanBodyBones.Neck, MsflowSkeleton.Neck },
                { HumanBodyBones.LeftShoulder, MsflowSkeleton.LeftCollar },
                { HumanBodyBones.RightShoulder, MsflowSkeleton.RightCollar },
                { HumanBodyBones.Head, MsflowSkeleton.Head },
                { HumanBodyBones.LeftUpperArm, MsflowSkeleton.LeftShoulder },
                { HumanBodyBones.RightUpperArm, MsflowSkeleton.RightShoulder },
                { HumanBodyBones.LeftLowerArm, MsflowSkeleton.LeftElbow },
                { HumanBodyBones.RightLowerArm, MsflowSkeleton.RightElbow },
                { HumanBodyBones.LeftHand, MsflowSkeleton.LeftWrist },
                { HumanBodyBones.RightHand, MsflowSkeleton.RightWrist },
            };

        /// <summary>
        /// Inverse of the bone table: the source joint a humanoid bone pins.
        /// Spine bones follow the anatomical spread (Spine→spine1, Chest→spine2,
        /// UpperChest→spine3) — when an avatar folds the chain (e.g. no Spine
        /// bone), re-locate the slot against the resolved mapping.
        /// </summary>
        public static bool TryGetJointForBone(HumanBodyBones bone, out int joint)
        {
            return JointForBone.TryGetValue(bone, out joint);
        }

        public static MsflowBoneMapping Resolve(Animator animator)
        {
            if (animator == null)
            {
                throw new ArgumentNullException(nameof(animator));
            }
            if (animator.avatar == null || !animator.avatar.isHuman)
            {
                throw new InvalidOperationException(
                    "MSFlow playback needs a Humanoid avatar on the Animator");
            }

            Transform Get(HumanBodyBones bone) => animator.GetBoneTransform(bone);

            var mapping = new MsflowBoneMapping
            {
                Bones =
                {
                    [MsflowSkeleton.Pelvis] = Get(HumanBodyBones.Hips),
                    [MsflowSkeleton.LeftHip] = Get(HumanBodyBones.LeftUpperLeg),
                    [MsflowSkeleton.RightHip] = Get(HumanBodyBones.RightUpperLeg),
                    [MsflowSkeleton.LeftKnee] = Get(HumanBodyBones.LeftLowerLeg),
                    [MsflowSkeleton.RightKnee] = Get(HumanBodyBones.RightLowerLeg),
                    [MsflowSkeleton.LeftAnkle] = Get(HumanBodyBones.LeftFoot),
                    [MsflowSkeleton.RightAnkle] = Get(HumanBodyBones.RightFoot),
                    [MsflowSkeleton.LeftFoot] = Get(HumanBodyBones.LeftToes),
                    [MsflowSkeleton.RightFoot] = Get(HumanBodyBones.RightToes),
                    [MsflowSkeleton.Neck] = Get(HumanBodyBones.Neck),
                    [MsflowSkeleton.Head] = Get(HumanBodyBones.Head),
                    [MsflowSkeleton.LeftCollar] = Get(HumanBodyBones.LeftShoulder),
                    [MsflowSkeleton.RightCollar] = Get(HumanBodyBones.RightShoulder),
                    [MsflowSkeleton.LeftShoulder] = Get(HumanBodyBones.LeftUpperArm),
                    [MsflowSkeleton.RightShoulder] = Get(HumanBodyBones.RightUpperArm),
                    [MsflowSkeleton.LeftElbow] = Get(HumanBodyBones.LeftLowerArm),
                    [MsflowSkeleton.RightElbow] = Get(HumanBodyBones.RightLowerArm),
                    [MsflowSkeleton.LeftWrist] = Get(HumanBodyBones.LeftHand),
                    [MsflowSkeleton.RightWrist] = Get(HumanBodyBones.RightHand),
                },
            };

            var spineSlots = SelectSpineChain(
                Get(HumanBodyBones.Spine), Get(HumanBodyBones.Chest), Get(HumanBodyBones.UpperChest));
            mapping.Bones[MsflowSkeleton.Spine1] = spineSlots[0];
            mapping.Bones[MsflowSkeleton.Spine2] = spineSlots[1];
            mapping.Bones[MsflowSkeleton.Spine3] = spineSlots[2];
            if (spineSlots[1] == null)
            {
                mapping.Warnings.Add(
                    "no second spine slot on this avatar; spine2 folds into the chain directions");
            }
            if (spineSlots[2] == null)
            {
                mapping.Warnings.Add(
                    "no third spine slot on this avatar; spine3 folds into the chain directions");
            }

            MarkDriven(mapping);
            ValidateRequired(mapping);
            if (mapping.Bones[MsflowSkeleton.LeftFoot] == null ||
                mapping.Bones[MsflowSkeleton.RightFoot] == null)
            {
                mapping.Warnings.Add("toe bones missing; feet end at the ankle direction");
            }
            if (mapping.Bones[MsflowSkeleton.Neck] == null)
            {
                mapping.Warnings.Add("neck bone missing; head rides the upper spine frame");
            }
            return mapping;
        }

        /// <summary>
        /// Spread spine1/spine2/spine3 over the avatar's available spine
        /// slots (Spine, Chest, UpperChest, anatomical order), anchored at
        /// both ends of the chain: two bones land on slots 0 and 2 so the
        /// lower and upper torso axes stay driven and the middle folds.
        /// Sources without a free slot fold into the chain directions of
        /// their driven neighbours.
        /// </summary>
        public static Transform[] SelectSpineChain(Transform spine, Transform chest, Transform upperChest)
        {
            var available = new List<Transform>(3);
            if (spine != null)
            {
                available.Add(spine);
            }
            if (chest != null)
            {
                available.Add(chest);
            }
            if (upperChest != null)
            {
                available.Add(upperChest);
            }
            var slots = new Transform[3];
            if (available.Count == 1)
            {
                slots[0] = available[0];
            }
            else if (available.Count > 1)
            {
                for (var i = 0; i < available.Count; i++)
                {
                    var slot = i * (slots.Length - 1) / (available.Count - 1);
                    slots[slot] = available[i];
                }
            }
            return slots;
        }

        /// <summary>
        /// Drive policy: every joint that has a bone is rotated by playback.
        /// Kept as a separate pure step so tests can pin it.
        /// </summary>
        public static void ApplyDrivenPolicy(MsflowBoneMapping mapping)
        {
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                mapping.Driven[joint] = mapping.Bones[joint] != null;
            }
        }

        private static void MarkDriven(MsflowBoneMapping mapping)
        {
            ApplyDrivenPolicy(mapping);
        }

        /// <summary>Throws when required humanoid bones are missing (with the list).</summary>
        public static void ValidateRequired(MsflowBoneMapping mapping)
        {
            var missing = new List<string>();
            foreach (var joint in RequiredJoints)
            {
                if (mapping.Bones[joint] == null)
                {
                    missing.Add($"{joint} {MsflowSkeleton.JointNames[joint]}");
                }
            }
            if (mapping.Bones[MsflowSkeleton.Spine1] == null &&
                mapping.Bones[MsflowSkeleton.Spine2] == null &&
                mapping.Bones[MsflowSkeleton.Spine3] == null)
            {
                missing.Add("any of spine1/spine2/spine3 (Spine/Chest/UpperChest)");
            }
            if (missing.Count > 0)
            {
                throw new InvalidOperationException(
                    "character is missing required humanoid bones: " + string.Join(", ", missing));
            }
        }
    }
}
