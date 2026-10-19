using System;
using UnityEngine;
using UnityEngine.UI;

namespace MSFlow.MotionBridge
{
    /// <summary>
    /// Minimal runtime demo: builds a small uGUI panel (prompt input, Generate
    /// button, progress/status readout), sends the prompt to the MSFlow
    /// runtime server, and plays the result on the target character.
    ///
    /// The demo scene is created via the "MSFlow > Create Basic Sample Scene"
    /// menu (Samples~/Basic/Editor/MsflowSampleSceneBuilder.cs). Assign any
    /// Humanoid character to the "Target Character" object, start the server
    /// (see unity_bridge/README.md), and press Play.
    /// </summary>
    public sealed class MsflowDemoController : MonoBehaviour
    {
        public string ServerUrl = "ws://127.0.0.1:8766";
        public string Prompt = "person waving hand";

        [Tooltip("Clip length in seconds. Shorter clips (2-3s) generate tamer, more gesture-like motion.")]
        public float DurationSeconds = 3f;

        [Tooltip("Send pelvis + ankle pins at the first and last frame (captured from the " +
                 "character's live stance) so the gesture stays grounded and local instead of " +
                 "a full-body performance.")]
        public bool PinStance = true;

        [Tooltip("Sampling steps for constraint runs (server default is 300; lower is faster).")]
        public int ConstraintSteps = 100;

        public MsflowMotionRequest Request { get; private set; }

        [SerializeField] private Text _status;
        [SerializeField] private Text _progress;
        [SerializeField] private Button _generateButton;
        [SerializeField] private InputField _promptInput;

        private void Awake()
        {
            Request = GetComponent<MsflowMotionRequest>();
            if (Request == null)
            {
                Request = gameObject.AddComponent<MsflowMotionRequest>();
            }
            Request.ServerUrl = ServerUrl;
            Request.GenerationProgress += OnProgress;
            Request.GenerationCompleted += OnCompleted;
            Request.GenerationFailed += OnFailed;
            BuildUi();
        }

        private void OnDestroy()
        {
            if (Request != null)
            {
                Request.GenerationProgress -= OnProgress;
                Request.GenerationCompleted -= OnCompleted;
                Request.GenerationFailed -= OnFailed;
            }
        }

        private void Start()
        {
            var character = Request.Target;
            if (character == null)
            {
                SetStatus("Create (or assign) a 'Target' object with a Humanoid Animator.");
                return;
            }
            if (character.GetComponent<Animator>() == null)
            {
                SetStatus("Target Character has no Animator — assign your Humanoid character here.");
            }
            else
            {
                SetStatus($"Ready. Server {ServerUrl}. Type a prompt and press Generate.");
            }
        }

        private void BuildUi()
        {
            if (_status == null)
            {
                Debug.Log("No Demo Canvas detected, creating...");

                var canvas = new GameObject("MsflowDemoCanvas", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
                canvas.transform.SetParent(transform, false);
                var canvasComponent = canvas.GetComponent<Canvas>();
                canvasComponent.renderMode = RenderMode.ScreenSpaceOverlay;
                canvas.GetComponent<CanvasScaler>().uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;

                if (UnityEngine.Object.FindObjectOfType<UnityEngine.EventSystems.EventSystem>() == null)
                {
                    new GameObject(
                        "MsflowDemoEventSystem",
                        typeof(UnityEngine.EventSystems.EventSystem),
                        typeof(UnityEngine.EventSystems.StandaloneInputModule));
                }

                var panel = CreatePanel(canvas.transform);
                _status = CreateText(panel, "Status", 20, TextAnchor.UpperLeft, new Vector2(-420, 60), new Vector2(820, 30));
                _progress = CreateText(panel, "Progress", 20, TextAnchor.UpperLeft, new Vector2(-420, 20), new Vector2(820, 30));
                _promptInput = CreateInput(panel, new Vector2(-420, -30), new Vector2(600, 40));
                _generateButton = CreateButton(panel, new Vector2(220, -30));
            }
            _promptInput.text = Prompt;
            _generateButton.onClick.AddListener(OnGenerateClicked);
        }

        private void OnGenerateClicked()
        {
            if (Request.IsGenerating)
            {
                SetStatus("Already generating — wait for the current motion.");
                return;
            }
            var prompt = _promptInput != null ? _promptInput.text : Prompt;
            if (string.IsNullOrWhiteSpace(prompt))
            {
                SetStatus("Type a prompt first, e.g. 'person waving hand'.");
                return;
            }
            _generateButton.interactable = false;
            SetProgress("sending request …");
            var controls = PinStance ? StanceControls() : null;
            Request.Generate(prompt, DurationSeconds, controls, PinStance ? (int?)ConstraintSteps : null);
        }

        /// <summary>
        /// Grounding pins (the webUI control schema): pelvis XZ and both
        /// ankles at the first and last frame, captured from the character's
        /// live stance via MsflowConstraintBuilder (positions-only pins).
        /// </summary>
        private System.Collections.Generic.List<MsflowControl> StanceControls()
        {
            if (Request.Target == null)
            {
                return null;
            }
            try
            {
                var builder = MsflowConstraintBuilder.ForDriver(Request.Target);
                builder.Role = "stance";
                foreach (var fraction in new[] { 0.0f, 1.0f })
                {
                    builder.PinPose(fraction, "x,z", HumanBodyBones.Hips);
                    builder.PinPose(fraction, "x,y,z", HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot);
                }
                return builder.Build();
            }
            catch (Exception exc)
            {
                SetStatus($"Stance pins unavailable: {exc.Message}");
                return null;
            }
        }

        private void OnProgress(string jobId, float fraction, string message)
        {
            SetProgress($"{fraction:P0}  {message}");
        }

        private void OnCompleted(MsflowMotion motion)
        {
            _generateButton.interactable = true;
            SetProgress("");
            SetStatus($"Playing {motion.FrameCount} frames ({motion.DurationSec:0.0}s @ {motion.Fps:0} fps).");
        }

        private void OnFailed(string message)
        {
            _generateButton.interactable = true;
            SetProgress("");
            SetStatus($"Failed: {message}");
        }

        private void SetStatus(string message)
        {
            if (_status != null)
            {
                _status.text = message;
            }
        }

        private void SetProgress(string message)
        {
            if (_progress != null)
            {
                _progress.text = message;
            }
        }

        // -- tiny uGUI construction helpers ---------------------------------

        private static RectTransform CreatePanel(Transform canvas)
        {
            var panel = new GameObject("Panel", typeof(Image));
            panel.transform.SetParent(canvas.transform, false);
            var rect = panel.GetComponent<RectTransform>();
            rect.anchorMin = new Vector2(0.5f, 0f);
            rect.anchorMax = new Vector2(0.5f, 0f);
            rect.pivot = new Vector2(0.5f, 0f);
            rect.anchoredPosition = new Vector2(0f, 8f);
            rect.sizeDelta = new Vector2(860f, 130f);
            panel.GetComponent<Image>().color = new Color(0f, 0f, 0f, 0.55f);
            return rect;
        }

        private static Font BuiltinFont()
        {
            try
            {
                return Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            }
            catch (Exception)
            {
                return Resources.GetBuiltinResource<Font>("Arial.ttf");
            }
        }

        private static Text CreateText(
            Transform parent, string name, int size, TextAnchor anchor, Vector2 position, Vector2 sizeDelta)
        {
            var go = new GameObject(name, typeof(Text));
            go.transform.SetParent(parent, false);
            var text = go.GetComponent<Text>();
            text.font = BuiltinFont();
            text.fontSize = size;
            text.alignment = anchor;
            text.color = Color.white;
            text.supportRichText = false;
            var rect = text.rectTransform;
            rect.anchorMin = new Vector2(0.5f, 0.5f);
            rect.anchorMax = new Vector2(0.5f, 0.5f);
            rect.anchoredPosition = position;
            rect.sizeDelta = sizeDelta;
            return text;
        }

        private static InputField CreateInput(Transform parent, Vector2 position, Vector2 sizeDelta)
        {
            var root = new GameObject("PromptInput", typeof(Image));
            root.transform.SetParent(parent, false);
            root.GetComponent<Image>().color = new Color(1f, 1f, 1f, 0.12f);
            var rootRect = root.GetComponent<RectTransform>();
            rootRect.anchorMin = rootRect.anchorMax = new Vector2(0.5f, 0.5f);
            rootRect.anchoredPosition = position;
            rootRect.sizeDelta = sizeDelta;

            var textObject = new GameObject("Text", typeof(Text));
            textObject.transform.SetParent(root.transform, false);
            var text = textObject.GetComponent<Text>();
            text.font = BuiltinFont();
            text.fontSize = 22;
            text.color = Color.white;
            text.supportRichText = false;
            var textRect = text.rectTransform;
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.offsetMin = new Vector2(10f, 4f);
            textRect.offsetMax = new Vector2(-10f, -4f);

            var placeholderObject = new GameObject("Placeholder", typeof(Text));
            placeholderObject.transform.SetParent(root.transform, false);
            var placeholder = placeholderObject.GetComponent<Text>();
            placeholder.font = BuiltinFont();
            placeholder.fontSize = 22;
            placeholder.color = new Color(1f, 1f, 1f, 0.4f);
            placeholder.text = "person waving hand";
            placeholder.supportRichText = false;
            var placeholderRect = placeholder.rectTransform;
            placeholderRect.anchorMin = Vector2.zero;
            placeholderRect.anchorMax = Vector2.one;
            placeholderRect.offsetMin = new Vector2(10f, 4f);
            placeholderRect.offsetMax = new Vector2(-10f, -4f);

            var input = root.AddComponent<InputField>();
            input.textComponent = text;
            input.placeholder = placeholder;
            return input;
        }

        private static Button CreateButton(Transform parent, Vector2 position)
        {
            var go = new GameObject("GenerateButton", typeof(Image), typeof(Button));
            go.transform.SetParent(parent, false);
            var image = go.GetComponent<Image>();
            image.color = new Color(0.2f, 0.55f, 0.95f);
            var rect = go.GetComponent<RectTransform>();
            rect.anchorMin = rect.anchorMax = new Vector2(0.5f, 0.5f);
            rect.anchoredPosition = position;
            rect.sizeDelta = new Vector2(200f, 40f);

            var labelObject = new GameObject("Label", typeof(Text));
            labelObject.transform.SetParent(go.transform, false);
            var label = labelObject.GetComponent<Text>();
            label.font = BuiltinFont();
            label.fontSize = 22;
            label.alignment = TextAnchor.MiddleCenter;
            label.color = Color.white;
            label.text = "Generate";
            label.supportRichText = false;
            var labelRect = label.rectTransform;
            labelRect.anchorMin = Vector2.zero;
            labelRect.anchorMax = Vector2.one;
            labelRect.offsetMin = Vector2.zero;
            labelRect.offsetMax = Vector2.zero;
            return go.GetComponent<Button>();
        }
    }
}
