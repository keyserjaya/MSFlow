using System;
using NUnit.Framework;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace MSFlow.MotionBridge.Tests
{
    public sealed class MsflowJsonTests
    {
        private const string ValidDoc = @"
{
  ""format"": ""msflow-motion"",
  ""version"": 1,
  ""fps"": 20,
  ""n_frames"": 2,
  ""duration_sec"": 0.1,
  ""joints"": [""pelvis"", ""left_hip"", ""right_hip"", ""spine1"", ""left_knee"",
    ""right_knee"", ""spine2"", ""left_ankle"", ""right_ankle"", ""spine3"",
    ""left_foot"", ""right_foot"", ""neck"", ""left_collar"", ""right_collar"",
    ""head"", ""left_shoulder"", ""right_shoulder"", ""left_elbow"",
    ""right_elbow"", ""left_wrist"", ""right_wrist""],
  ""parents"": [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19],
  ""positions"": [
    [[0.0, 0.95, 0.0], [-0.09, 0.97, 0.0], [0.09, 0.97, 0.0], [0.0, 1.02, 0.0],
     [-0.09, 0.55, 0.02], [0.09, 0.55, 0.02], [0.0, 1.10, 0.0], [-0.10, 0.09, 0.05],
     [0.10, 0.09, 0.05], [0.0, 1.20, 0.0], [-0.12, 0.03, 0.11], [0.12, 0.03, 0.11],
     [0.0, 1.42, 0.02], [-0.06, 1.44, 0.02], [0.06, 1.44, 0.02], [0.0, 1.62, 0.03],
     [-0.20, 1.40, 0.02], [0.20, 1.40, 0.02], [-0.20, 1.12, 0.02], [0.20, 1.12, 0.02],
     [-0.20, 0.90, 0.05], [0.20, 0.90, 0.05]],
    [[0.0, 0.95, 0.01], [-0.09, 0.97, 0.01], [0.09, 0.97, 0.01], [0.0, 1.02, 0.01],
     [-0.09, 0.55, 0.03], [0.09, 0.55, 0.03], [0.0, 1.10, 0.01], [-0.10, 0.09, 0.06],
     [0.10, 0.09, 0.06], [0.0, 1.20, 0.01], [-0.12, 0.03, 0.12], [0.12, 0.03, 0.12],
     [0.0, 1.42, 0.03], [-0.06, 1.44, 0.03], [0.06, 1.44, 0.03], [0.0, 1.62, 0.04],
     [-0.20, 1.40, 0.03], [0.20, 1.40, 0.03], [-0.20, 1.12, 0.03], [0.20, 1.12, 0.03],
     [-0.20, 0.90, 0.06], [0.20, 0.90, 0.06]]
  ],
  ""controls"": [
    { ""frame"": 1, ""joints"": [20], ""axes"": ""x,y,z"", ""values"": [[0.5, 1.2, 0.3]] }
  ],
  ""meta"": { ""prompt"": ""person waving hand"" }
}
";

        [Test]
        public void ParsesValidDocument()
        {
            var motion = MsflowMotion.FromJson(ValidDoc);
            Assert.AreEqual(20f, motion.Fps);
            Assert.AreEqual(2, motion.FrameCount);
            Assert.AreEqual(0.1f, motion.DurationSec, 1e-4f);
            Assert.AreEqual(MsflowSkeleton.JointCount, motion.JointNames.Length);
            Assert.AreEqual(22, motion.Parents.Length);
            Assert.AreEqual(-1, motion.Parents[0]);
            Assert.AreEqual(9, motion.Parents[12]);
            Assert.AreEqual(1, motion.Controls.Count);
            Assert.AreEqual(20, motion.Controls[0].Joints[0]);
            Assert.AreEqual(0.95f, motion.GetJoint(0, MsflowSkeleton.Pelvis).y, 1e-5f);
            // interpolated halfway between frame 0 and 1 (pelvis z drifts 0 -> 0.01)
            Assert.AreEqual(0.95f, motion.GetJointInterpolated(0.025f, MsflowSkeleton.Pelvis).y, 1e-4f);
            Assert.AreEqual(0.005f, motion.GetJointInterpolated(0.025f, MsflowSkeleton.Pelvis).z, 1e-4f);
        }

        [Test]
        public void RejectsWrongFormat()
        {
            var doc = JObject.Parse(ValidDoc);
            doc["format"] = "other";
            var exc = Assert.Throws<MsflowMotionException>(
                () => MsflowMotion.FromJObject(doc));
            StringAssert.Contains("msflow-motion", exc.Message);
        }

        [Test]
        public void RejectsWrongVersion()
        {
            var doc = JObject.Parse(ValidDoc);
            doc["version"] = 2;
            Assert.Throws<MsflowMotionException>(() => MsflowMotion.FromJObject(doc));
        }

        [Test]
        public void RejectsJaggedFrame()
        {
            var doc = JObject.Parse(ValidDoc);
            ((JArray)doc["positions"])[0][3] = new JArray(0f, 1f);
            var exc = Assert.Throws<MsflowMotionException>(
                () => MsflowMotion.FromJObject(doc));
            StringAssert.Contains("3 floats", exc.Message);
        }

        [Test]
        public void RejectsBadParentOrder()
        {
            var doc = JObject.Parse(ValidDoc);
            ((JArray)doc["parents"])[5] = 9; // child index before parent is fine, but 9 >= 5
            var exc = Assert.Throws<MsflowMotionException>(
                () => MsflowMotion.FromJObject(doc));
            StringAssert.Contains("tree order", exc.Message);
        }

        [Test]
        public void RejectsNonFiniteCoordinates()
        {
            var doc = JObject.Parse(ValidDoc);
            ((JArray)doc["positions"])[1][20][0] = float.NaN;
            Assert.Throws<MsflowMotionException>(() => MsflowMotion.FromJObject(doc));
        }

        [Test]
        public void GenerateMessageBuildsProtocolV1Fields()
        {
            var client = new MsflowSocketClient("ws://127.0.0.1:8766");
            var request = new MsflowGenerateRequest
            {
                Prompt = "person waving hand",
                Seed = 1234,
                Controls = new System.Collections.Generic.List<MsflowControl>
                {
                    new MsflowControl
                    {
                        Frame = 20,
                        Joints = new[] { 20 },
                        Values = new[] { new[] { 0.5f, 1.2f, 0.3f } },
                    },
                },
            };
            var message = client.BuildGenerateMessage(request, "job-1");
            Assert.AreEqual("generate", message.Value<string>("type"));
            Assert.AreEqual("job-1", message.Value<string>("job_id"));
            Assert.AreEqual("person waving hand", message.Value<string>("prompt"));
            Assert.AreEqual(6f, message.Value<float?>("duration_sec") ?? 0f, 1e-5f);
            Assert.AreEqual(1234, message.Value<int?>("seed"));
            Assert.AreEqual(20, message["controls"][0]["frame"].Value<int>());
            Assert.IsNull(message["controls"][0]["frame_fraction"]);
        }
    }
}
