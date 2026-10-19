using UnityEngine;

namespace MSFlow.MotionBridge
{
    /// <summary>
    /// Debug overlay: draws the generated motion's joint positions (the exact
    /// points the retarget aims at) as markers that follow playback in real
    /// time. Attach next to a <see cref="MsflowPlaybackDriver"/> to compare
    /// the posed character against its source pose frame-for-frame — e.g. to
    /// tell a crunchy source clip from a retarget artifact.
    /// </summary>
    [AddComponentMenu("MSFlow/MSFlow Source Overlay")]
    public sealed class MsflowSourceOverlay : MonoBehaviour
    {
        [Tooltip("Driver to visualize. Empty = first driver on this object or in children.")]
        public MsflowPlaybackDriver Driver;

        [Min(0.001f)]
        [Tooltip("Marker diameter in metres.")]
        public float MarkerSize = 0.05f;

        public Color MarkerColor = new Color(0.2f, 0.85f, 1f, 0.9f);

        [Tooltip("Show the markers while playback is paused/stopped (at time 0).")]
        public bool ShowWhenIdle = false;

        private readonly Transform[] _markers = new Transform[MsflowSkeleton.JointCount];
        private Transform _container;

        private void Awake()
        {
            if (Driver == null)
            {
                Driver = GetComponent<MsflowPlaybackDriver>();
                if (Driver == null)
                {
                    Driver = GetComponentInChildren<MsflowPlaybackDriver>();
                }
            }
            _container = new GameObject("MsflowSourceMarkers").transform;
            _container.SetParent(transform, false);
            var material = new Material(Shader.Find("Unlit/Color"));
            material.color = MarkerColor;
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                var marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                marker.name = $"src_{joint:00}_{MsflowSkeleton.JointNames[joint]}";
                var collider = marker.GetComponent<Collider>();
                if (collider != null)
                {
                    Destroy(collider);
                }
                marker.GetComponent<Renderer>().sharedMaterial = material;
                marker.transform.SetParent(_container, false);
                marker.transform.localScale = Vector3.one * MarkerSize;
                _markers[joint] = marker.transform;
            }
        }

        private void LateUpdate()
        {
            if (Driver == null)
            {
                return;
            }
            var playing = Driver.IsPlaying;
            _container.gameObject.SetActive(playing || ShowWhenIdle);
            if (!playing && !ShowWhenIdle)
            {
                return;
            }
            var time = playing ? Driver.PlaybackTime : 0f;
            for (var joint = 0; joint < MsflowSkeleton.JointCount; joint++)
            {
                if (Driver.TryGetSourcePosition(joint, time, out var world))
                {
                    _markers[joint].position = world;
                }
            }
        }

        private void OnDestroy()
        {
            if (_container != null)
            {
                Destroy(_container.gameObject);
            }
        }
    }
}
