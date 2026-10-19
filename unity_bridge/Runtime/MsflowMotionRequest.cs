using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace MSFlow.MotionBridge
{
    /// <summary>
    /// High-level Unity-facing facade: send a prompt (optionally with sparse
    /// controls) to the MSFlow runtime server, stream progress, and hand the
    /// finished motion to a <see cref="MsflowPlaybackDriver"/>.
    ///
    ///     var motion = await request.GenerateAsync("person waving hand");
    ///
    /// Callers that prefer callbacks can subscribe to GenerationProgress /
    /// GenerationCompleted / GenerationFailed instead. The underlying socket
    /// client's callbacks are marshalled to the main thread via PumpMain().
    /// </summary>
    [AddComponentMenu("MSFlow/MSFlow Motion Request")]
    public sealed class MsflowMotionRequest : MonoBehaviour
    {
        [Tooltip("ws:// URL of the MSFlow runtime server (runs in WSL2).")]
        public string ServerUrl = "ws://127.0.0.1:8766";

        [Tooltip("Play finished motions on this driver. Empty = first driver in children.")]
        public MsflowPlaybackDriver Target;

        public bool AutoPlay = true;

        [Tooltip("Snapshot the target driver's live pose at the clip start (frame 0) and pin it " +
                 "under any caller controls — the generated motion then starts from the pose the " +
                 "character is actually in. Caller pins win on duplicates.")]
        public bool PinStartPose = false;

        /// <summary>
        /// PinStartPose snapshot source; defaults to the target driver's live
        /// pose via MsflowConstraintBuilder.ForDriver. Tests and custom rigs
        /// can substitute their own builder (not serialized).
        /// </summary>
        [NonSerialized]
        public Func<MsflowConstraintBuilder> StartPoseSnapshot;

        public MsflowSocketClient Client { get; private set; }
        public bool IsGenerating { get; private set; }

        public event Action<string, float, string> GenerationProgress;
        public event Action<MsflowMotion> GenerationCompleted;
        public event Action<string> GenerationFailed;

        private string _activeJob;
        private TaskCompletionSource<MsflowMotion> _completion;

        private void Awake()
        {
            Client = new MsflowSocketClient(ServerUrl);
            Client.Progress += OnClientProgress;
            Client.Result += OnClientResult;
            Client.Error += OnClientError;
            Client.Disconnected += OnClientDisconnected;
            if (Target == null)
            {
                Target = GetComponentInChildren<MsflowPlaybackDriver>();
            }
        }

        private void Update()
        {
            Client?.PumpMain();
        }

        private void OnDestroy()
        {
            if (Client != null)
            {
                Client.Progress -= OnClientProgress;
                Client.Result -= OnClientResult;
                Client.Error -= OnClientError;
                Client.Disconnected -= OnClientDisconnected;
                Client.Dispose();
                Client = null;
            }
        }

        /// <summary>
        /// Generate one motion from a text prompt (and optional sparse
        /// controls). Completes with the parsed motion, or throws with the
        /// server's error message.
        /// </summary>
        public async Task<MsflowMotion> GenerateAsync(
            string prompt,
            float durationSec = 6f,
            IList<MsflowControl> controls = null,
            int? numSteps = null,
            CancellationToken ct = default)
        {
            if (IsGenerating)
            {
                throw new InvalidOperationException("a generation is already in flight");
            }
            if (Client == null)
            {
                throw new InvalidOperationException("client destroyed");
            }

            if (!Client.IsConnected)
            {
                await Client.ConnectAsync(ct).ConfigureAwait(true);
            }
            if (ct.IsCancellationRequested)
            {
                throw new OperationCanceledException(ct);
            }

            var effectiveControls = PinStartPose ? ComposeControls(controls) : controls;

            IsGenerating = true;
            _completion = new TaskCompletionSource<MsflowMotion>(
                TaskCreationOptions.RunContinuationsAsynchronously);
            var request = new MsflowGenerateRequest
            {
                Prompt = prompt,
                DurationSec = durationSec,
                Controls = effectiveControls,
                NumSteps = numSteps,
            };
            try
            {
                _activeJob = await Client.GenerateAsync(request, ct).ConfigureAwait(true);
            }
            catch (Exception)
            {
                IsGenerating = false;
                throw;
            }

            var registration = ct.Register(() =>
            {
                if (_activeJob != null && Client != null)
                {
                    _ = Client.CancelAsync(_activeJob);
                }
                FailGeneration("cancelled");
            });
            try
            {
                return await _completion.Task.ConfigureAwait(true);
            }
            finally
            {
                registration.Dispose();
                _activeJob = null;
            }
        }

        /// <summary>Fire-and-forget variant for UI wiring.</summary>
        public async void Generate(
            string prompt, float durationSec = 6f,
            IList<MsflowControl> controls = null, int? numSteps = null)
        {
            try
            {
                await GenerateAsync(prompt, durationSec, controls, numSteps).ConfigureAwait(true);
            }
            catch (Exception exc)
            {
                // GenerationFailed already fired for protocol-level errors;
                // surface client-side setup failures here.
                if (IsGenerating)
                {
                    FailGeneration(exc.Message);
                }
                else
                {
                    GenerationFailed?.Invoke(exc.Message);
                }
            }
        }

        /// <summary>
        /// Generate with constraint-builder controls. Throughout pins expand
        /// against the requested clip length (FrameCount = duration × 20 fps).
        /// </summary>
        public Task<MsflowMotion> GenerateAsync(
            string prompt,
            float durationSec,
            MsflowConstraintBuilder builder,
            int? numSteps = null,
            CancellationToken ct = default)
        {
            if (builder == null)
            {
                throw new ArgumentNullException(nameof(builder));
            }
            builder.FrameCount = System.Math.Max(
                2, (int)System.Math.Round(durationSec * MsflowConstraintBuilder.MotionFrameRate));
            return GenerateAsync(prompt, durationSec, builder.Build(), numSteps, ct);
        }

        /// <summary>Fire-and-forget variant of the builder overload.</summary>
        public async void Generate(
            string prompt, float durationSec, MsflowConstraintBuilder builder, int? numSteps = null)
        {
            try
            {
                await GenerateAsync(prompt, durationSec, builder, numSteps).ConfigureAwait(true);
            }
            catch (Exception exc)
            {
                if (IsGenerating)
                {
                    FailGeneration(exc.Message);
                }
                else
                {
                    GenerationFailed?.Invoke(exc.Message);
                }
            }
        }

        /// <summary>
        /// The wire controls one request actually sends: with
        /// <see cref="PinStartPose"/> set, the target driver's live pose is
        /// snapshotted at frame 0 and merged under <paramref name="callerControls"/>
        /// (caller pins win on duplicates). Without it the caller list passes
        /// through unchanged.
        /// </summary>
        public List<MsflowControl> ComposeControls(IList<MsflowControl> callerControls)
        {
            if (!PinStartPose)
            {
                return callerControls == null
                    ? new List<MsflowControl>()
                    : new List<MsflowControl>(callerControls);
            }
            if (Target == null && StartPoseSnapshot == null)
            {
                throw new InvalidOperationException(
                    "PinStartPose needs a target driver with a Humanoid animator");
            }
            var snapshot = StartPoseSnapshot != null
                ? StartPoseSnapshot()
                : MsflowConstraintBuilder.ForDriver(Target);
            snapshot.PinCurrentPoseAtStart();
            return MsflowConstraintBuilder.MergeControls(snapshot.Build(), callerControls);
        }

        private void OnClientProgress(string jobId, float fraction, string message)
        {
            if (jobId != _activeJob)
            {
                return;
            }
            GenerationProgress?.Invoke(jobId, fraction, message);
        }

        private void OnClientResult(string jobId, MsflowMotion motion, string raw)
        {
            if (jobId != _activeJob)
            {
                return;
            }
            IsGenerating = false;
            GenerationCompleted?.Invoke(motion);
            if (AutoPlay && Target != null)
            {
                Target.Play(motion);
            }
            _completion?.TrySetResult(motion);
        }

        private void OnClientError(string jobId, string message)
        {
            if (jobId == _activeJob)
            {
                FailGeneration(message);
            }
        }

        private void OnClientDisconnected(string reason)
        {
            if (IsGenerating)
            {
                FailGeneration($"disconnected: {reason}");
            }
        }

        private void FailGeneration(string message)
        {
            IsGenerating = false;
            GenerationFailed?.Invoke(message);
            _completion?.TrySetException(new MsflowMotionException(message));
            _completion = null;
        }
    }
}
