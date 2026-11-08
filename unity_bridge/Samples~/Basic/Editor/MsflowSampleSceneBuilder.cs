/* Disabled due to demo scene already exist
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace MSFlow.MotionBridge
{
    /// <summary>
    /// Creates the Basic sample scene programmatically (no shipped scene
    /// asset to break across Unity versions): ground plane, demo controller,
    /// and an empty "Target Character" slot for your own Humanoid.
    /// </summary>
    public static class MsflowSampleSceneBuilder
    {
        [MenuItem("MSFlow/Create Basic Sample Scene")]
        public static void CreateSampleScene()
        {
            var scene = EditorSceneManager.NewScene(
                NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.position = Vector3.zero;
            ground.transform.localScale = new Vector3(10f, 1f, 10f);

            var target = new GameObject("Target Character");
            target.transform.position = Vector3.zero;
            target.AddComponent<MsflowPlaybackDriver>();
            var request = target.AddComponent<MsflowMotionRequest>();

            var demo = new GameObject("MSFlow Demo");
            var controller = demo.AddComponent<MsflowDemoController>();
            controller.ServerUrl = request.ServerUrl;

            EditorSceneManager.SaveScene(scene, "Assets/MSFlowDemo.unity");
            Debug.Log(
                "[msflow] Sample scene saved to Assets/MSFlowDemo.unity. " +
                "Drag your Humanoid character under 'Target Character' (or replace it), " +
                "start the runtime server (see unity_bridge/README.md), then press Play.");
        }
    }
}
*/