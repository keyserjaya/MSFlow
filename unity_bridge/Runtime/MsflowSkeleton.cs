using System.Collections.Generic;

namespace MSFlow.MotionBridge
{
    /// <summary>
    /// HumanML3D (t2m) 22-joint layout constants, mirroring webui/skeleton.py.
    /// Joints are world XYZ positions in metres, Y-up, character facing +Z,
    /// character-right at +X (the same convention Unity characters use).
    /// </summary>
    public static class MsflowSkeleton
    {
        public const int JointCount = 22;

        public const int Pelvis = 0;
        public const int LeftHip = 1;
        public const int RightHip = 2;
        public const int Spine1 = 3;
        public const int LeftKnee = 4;
        public const int RightKnee = 5;
        public const int Spine2 = 6;
        public const int LeftAnkle = 7;
        public const int RightAnkle = 8;
        public const int Spine3 = 9;
        public const int LeftFoot = 10;
        public const int RightFoot = 11;
        public const int Neck = 12;
        public const int LeftCollar = 13;
        public const int RightCollar = 14;
        public const int Head = 15;
        public const int LeftShoulder = 16;
        public const int RightShoulder = 17;
        public const int LeftElbow = 18;
        public const int RightElbow = 19;
        public const int LeftWrist = 20;
        public const int RightWrist = 21;

        public static readonly string[] JointNames =
        {
            "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
            "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
            "neck", "left_collar", "right_collar", "head", "left_shoulder",
            "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        };

        /// <summary>Parent joint index per joint (-1 for the pelvis root).</summary>
        public static readonly int[] Parents =
        {
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19,
        };

        /// <summary>
        /// Chain continuations that beat the tree's parenthood: the pelvis
        /// continues into the spine (not the leg) and the neck continues into
        /// the head (not the collars) — same as blender retarget.py.
        /// </summary>
        private static readonly Dictionary<int, int> SectionOverrides = new Dictionary<int, int>
        {
            { Pelvis, Spine1 },
            { Neck, Head },
        };

        /// <summary>
        /// Per joint, the child joint along its anatomical continuation chain
        /// (used as the bone's +Y direction target); -1 for chain ends.
        /// </summary>
        public static int[] ChainSuccessors()
        {
            var children = new List<int>[JointCount];
            for (var i = 0; i < JointCount; i++)
            {
                children[i] = new List<int>();
            }
            for (var joint = 0; joint < JointCount; joint++)
            {
                var parent = Parents[joint];
                if (parent >= 0)
                {
                    children[parent].Add(joint);
                }
            }
            var successors = new int[JointCount];
            for (var joint = 0; joint < JointCount; joint++)
            {
                if (SectionOverrides.TryGetValue(joint, out var overridden))
                {
                    successors[joint] = overridden;
                }
                else if (children[joint].Count > 0)
                {
                    successors[joint] = children[joint][0]; // min child
                }
                else
                {
                    successors[joint] = -1;
                }
            }
            return successors;
        }

        /// <summary>
        /// Converts one joint position from the generated data's convention
        /// into Unity's. The data is X-mirrored relative to Unity (measured:
        /// its hip-line x trunk cross product points opposite the toe-forward
        /// direction, dot = -1.00 across clips), which flips both the derived
        /// facing and the left/right limbs. Negating X restores Unity's
        /// convention (facing +Z, character-right at +X, labels matching
        /// HumanBodyBones) with forward preserved.
        /// </summary>
        public static UnityEngine.Vector3 FromDataConvention(UnityEngine.Vector3 data)
        {
            return new UnityEngine.Vector3(-data.x, data.y, data.z);
        }

        /// <summary>
        /// A plausible standing pose [22 x 3] in world XYZ (metres), matching
        /// webui/skeleton.py canonical_pose(); serves as the source rest pose.
        /// </summary>
        public static float[][] CanonicalPose()
        {
            float[][] pose =
            {
                new[] { 0.00f, 0.95f, 0.00f }, // 0 pelvis
                new[] { -0.09f, 0.97f, 0.00f }, // 1 left_hip
                new[] { 0.09f, 0.97f, 0.00f }, // 2 right_hip
                new[] { 0.00f, 1.02f, 0.00f }, // 3 spine1
                new[] { -0.09f, 0.55f, 0.02f }, // 4 left_knee
                new[] { 0.09f, 0.55f, 0.02f }, // 5 right_knee
                new[] { 0.00f, 1.10f, 0.00f }, // 6 spine2
                new[] { -0.10f, 0.09f, 0.05f }, // 7 left_ankle
                new[] { 0.10f, 0.09f, 0.05f }, // 8 right_ankle
                new[] { 0.00f, 1.20f, 0.00f }, // 9 spine3
                new[] { -0.12f, 0.03f, 0.11f }, // 10 left_foot
                new[] { 0.12f, 0.03f, 0.11f }, // 11 right_foot
                new[] { 0.00f, 1.42f, 0.02f }, // 12 neck
                new[] { -0.06f, 1.44f, 0.02f }, // 13 left_collar
                new[] { 0.06f, 1.44f, 0.02f }, // 14 right_collar
                new[] { 0.00f, 1.62f, 0.03f }, // 15 head
                new[] { -0.20f, 1.40f, 0.02f }, // 16 left_shoulder
                new[] { 0.20f, 1.40f, 0.02f }, // 17 right_shoulder
                new[] { -0.20f, 1.12f, 0.02f }, // 18 left_elbow
                new[] { 0.20f, 1.12f, 0.02f }, // 19 right_elbow
                new[] { -0.20f, 0.90f, 0.05f }, // 20 left_wrist
                new[] { 0.20f, 0.90f, 0.05f }, // 21 right_wrist
            };
            return pose;
        }
    }
}
