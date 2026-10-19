using System;
using System.Collections.Generic;
using System.Globalization;
using Newtonsoft.Json.Linq;

namespace MSFlow.MotionBridge
{
    /// <summary>One sparse-control entry (the webui spec-loader schema).</summary>
    public sealed class MsflowControl
    {
        public int? Frame;
        public float? FrameFraction;
        public int[] Joints = Array.Empty<int>();
        public string Axes = "x,y,z";
        public float[][] Values = Array.Empty<float[]>();
        public string Stage = "";
        public string Role = "";

        public JObject ToJObject()
        {
            var o = new JObject();
            if (Frame.HasValue)
            {
                o["frame"] = Frame.Value;
            }
            if (FrameFraction.HasValue)
            {
                o["frame_fraction"] = FrameFraction.Value;
            }
            o["joints"] = new JArray(Joints);
            if (!string.IsNullOrEmpty(Axes))
            {
                o["axes"] = Axes;
            }
            var values = new JArray();
            foreach (var row in Values)
            {
                values.Add(new JArray(row));
            }
            o["values"] = values;
            if (!string.IsNullOrEmpty(Stage))
            {
                o["stage"] = Stage;
            }
            if (!string.IsNullOrEmpty(Role))
            {
                o["role"] = Role;
            }
            return o;
        }

        public static MsflowControl FromJObject(JObject o)
        {
            var control = new MsflowControl();
            if (o["frame"] is JValue frame)
            {
                control.Frame = Convert.ToInt32(frame.Value, CultureInfo.InvariantCulture);
            }
            if (o["frame_fraction"] is JValue fraction)
            {
                control.FrameFraction = Convert.ToSingle(fraction.Value, CultureInfo.InvariantCulture);
            }
            if (o["joints"] is JArray joints)
            {
                var ids = new List<int>();
                foreach (var item in joints)
                {
                    ids.Add(Convert.ToInt32(((JValue)item).Value, CultureInfo.InvariantCulture));
                }
                control.Joints = ids.ToArray();
            }
            control.Axes = o.Value<string>("axes") ?? "x,y,z";
            if (o["values"] is JArray values)
            {
                var rows = new List<float[]>();
                foreach (var row in values)
                {
                    var floats = new List<float>();
                    foreach (var item in (JArray)row)
                    {
                        floats.Add(Convert.ToSingle(((JValue)item).Value, CultureInfo.InvariantCulture));
                    }
                    rows.Add(floats.ToArray());
                }
                control.Values = rows.ToArray();
            }
            control.Stage = o.Value<string>("stage") ?? "";
            control.Role = o.Value<string>("role") ?? "";
            return control;
        }
    }

    /// <summary>
    /// A parsed <c>msflow-motion</c> v1 document: 22-joint world positions at
    /// a fixed frame rate. Mirrors blender_addon/msflow_bridge/io_msflow.py.
    /// </summary>
    public sealed class MsflowMotion
    {
        public float Fps { get; private set; }
        public int FrameCount { get; private set; }
        public float DurationSec => FrameCount / Fps;
        public string[] JointNames { get; private set; }
        public int[] Parents { get; private set; }

        /// <summary>[frame][joint][xyz] world positions, metres, Y-up.</summary>
        public float[][][] Positions { get; private set; }

        public IReadOnlyList<MsflowControl> Controls => _controls;
        private readonly List<MsflowControl> _controls = new List<MsflowControl>();

        public static MsflowMotion FromJson(string json)
        {
            JObject document;
            try
            {
                document = JObject.Parse(json);
            }
            catch (Exception exc)
            {
                throw new MsflowMotionException($"motion payload is not valid JSON: {exc.Message}");
            }
            return FromJObject(document);
        }

        public static MsflowMotion FromJObject(JObject document)
        {
            var format = document.Value<string>("format");
            if (format != "msflow-motion")
            {
                throw new MsflowMotionException($"expected format 'msflow-motion', got '{format}'");
            }
            if (document.Value<int?>("version") != 1)
            {
                throw new MsflowMotionException("unsupported msflow-motion version (want 1)");
            }
            var fps = document.Value<float?>("fps") ?? 0f;
            if (fps <= 0f)
            {
                throw new MsflowMotionException($"fps must be positive, got {fps}");
            }

            var positions = ParsePositions(document["positions"] as JArray);
            var frameCount = positions.Length;

            var names = new string[MsflowSkeleton.JointCount];
            for (var i = 0; i < names.Length; i++)
            {
                names[i] = MsflowSkeleton.JointNames[i];
            }
            if (document["joints"] is JArray jointsArray && jointsArray.Count > 0)
            {
                if (jointsArray.Count != MsflowSkeleton.JointCount)
                {
                    throw new MsflowMotionException(
                        $"joints must list {MsflowSkeleton.JointCount} names, got {jointsArray.Count}");
                }
                for (var i = 0; i < jointsArray.Count; i++)
                {
                    names[i] = jointsArray[i].Value<string>();
                }
            }

            var parents = ParseParents(document["parents"] as JArray);

            var motion = new MsflowMotion
            {
                Fps = fps,
                FrameCount = frameCount,
                JointNames = names,
                Parents = parents,
                Positions = positions,
            };

            if (document["controls"] is JArray controls)
            {
                foreach (var entry in controls)
                {
                    if (entry is JObject controlObject)
                    {
                        motion._controls.Add(MsflowControl.FromJObject(controlObject));
                    }
                }
            }
            return motion;
        }

        private static float[][][] ParsePositions(JArray array)
        {
            if (array == null || array.Count == 0)
            {
                throw new MsflowMotionException("positions must be a non-empty [T][22][3] array");
            }
            var frames = new float[array.Count][][];
            for (var t = 0; t < array.Count; t++)
            {
                if (!(array[t] is JArray joints) || joints.Count != MsflowSkeleton.JointCount)
                {
                    throw new MsflowMotionException(
                        $"frame {t} must contain {MsflowSkeleton.JointCount} joints");
                }
                var frame = new float[MsflowSkeleton.JointCount][];
                for (var j = 0; j < joints.Count; j++)
                {
                    if (!(joints[j] is JArray xyz) || xyz.Count != 3)
                    {
                        throw new MsflowMotionException(
                            $"frame {t} joint {j} must contain 3 floats");
                    }
                    var row = new float[3];
                    for (var axis = 0; axis < 3; axis++)
                    {
                        var value = xyz[axis].Value<float?>();
                        if (value == null || float.IsNaN(value.Value) || float.IsInfinity(value.Value))
                        {
                            throw new MsflowMotionException(
                                $"frame {t} joint {j} axis {axis} is not a finite float");
                        }
                        row[axis] = value.Value;
                    }
                    frame[j] = row;
                }
                frames[t] = frame;
            }
            return frames;
        }

        private static int[] ParseParents(JArray array)
        {
            if (array == null)
            {
                // fall back to the canonical tree when the doc omits parents
                return (int[])MsflowSkeleton.Parents.Clone();
            }
            if (array.Count != MsflowSkeleton.JointCount)
            {
                throw new MsflowMotionException(
                    $"parents must list {MsflowSkeleton.JointCount} entries, got {array.Count}");
            }
            var parents = new int[MsflowSkeleton.JointCount];
            for (var i = 0; i < array.Count; i++)
            {
                parents[i] = array[i].Value<int>();
                if (parents[i] >= i)
                {
                    throw new MsflowMotionException(
                        $"parents[{i}] must be smaller than the joint index (tree order)");
                }
            }
            if (parents[0] != -1)
            {
                throw new MsflowMotionException("parents[0] must be -1 (the pelvis is the root)");
            }
            return parents;
        }

        /// <summary>World position of one joint at an integer frame.</summary>
        public UnityEngine.Vector3 GetJoint(int frame, int joint)
        {
            var row = Positions[frame][joint];
            return new UnityEngine.Vector3(row[0], row[1], row[2]);
        }

        /// <summary>Linearly interpolated joint position at a time in seconds.</summary>
        public UnityEngine.Vector3 GetJointInterpolated(float time, int joint)
        {
            var frameTime = ClampTime(time) * Fps;
            var i0 = (int)frameTime;
            var i1 = Math.Min(i0 + 1, FrameCount - 1);
            var frac = frameTime - i0;
            var a = GetJoint(i0, joint);
            var b = GetJoint(i1, joint);
            return UnityEngine.Vector3.LerpUnclamped(a, b, frac);
        }

        public float ClampTime(float time)
        {
            var duration = DurationSec;
            if (duration <= 0f)
            {
                return 0f;
            }
            return UnityEngine.Mathf.Clamp(time, 0f, duration);
        }
    }

    /// <summary>Raised for schema violations in an msflow-motion payload.</summary>
    public sealed class MsflowMotionException : Exception
    {
        public MsflowMotionException(string message) : base(message)
        {
        }
    }
}
