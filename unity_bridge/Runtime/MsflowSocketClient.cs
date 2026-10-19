using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

namespace MSFlow.MotionBridge
{
    /// <summary>Server hello payload (protocol capabilities).</summary>
    public struct MsflowServerCapabilities
    {
        public int Protocol;
        public float Fps;
        public int? MaxLength;
        public string Model;
        public string Ckpt;
    }

    /// <summary>One generate request (client-side).</summary>
    public sealed class MsflowGenerateRequest
    {
        public string Prompt = "";
        public float DurationSec = 6f;
        public int? NFrames;
        public int? Seed;
        public float? Cfg;
        public int? NumSteps;
        public IList<MsflowControl> Controls;
    }

    /// <summary>
    /// WebSocket client for the MSFlow runtime server (protocol v1).
    ///
    /// Plain C# (no MonoBehaviour): construct on the Unity main thread, call
    /// <see cref="PumpMain"/> once per frame from Update, and consume the
    /// events — they always fire on the main thread. On disconnect it
    /// reconnects automatically and re-fetches unfinished jobs.
    /// </summary>
    public sealed class MsflowSocketClient : IDisposable
    {
        public string Url { get; }
        public bool AutoReconnect = true;
        public float ReconnectDelaySeconds = 1f;

        public bool IsConnected => Volatile.Read(ref _state) == StateConnected;
        public bool IsConnecting => Volatile.Read(ref _state) == StateConnecting;

        public event Action Connected;
        public event Action<string> Disconnected;
        public event Action<MsflowServerCapabilities> Hello;
        public event Action<string, float, string> Progress;
        public event Action<string, MsflowMotion, string> Result;
        public event Action<string, string> Error;

        private const int StateIdle = 0;
        private const int StateConnecting = 1;
        private const int StateConnected = 2;
        private const int StateClosed = 3;

        private readonly ConcurrentQueue<Action> _mainThreadQueue = new ConcurrentQueue<Action>();
        private readonly Dictionary<string, MsflowGenerateRequest> _pending = new Dictionary<string, MsflowGenerateRequest>();
        private readonly object _gate = new object();
        private readonly SemaphoreSlim _sendLock = new SemaphoreSlim(1, 1);

        private ClientWebSocket _socket;
        private CancellationTokenSource _cts;
        private int _state = StateIdle;
        private bool _disposed;
        private TaskCompletionSource<MsflowServerCapabilities> _helloWaiter;
        private Task _sessionTask;

        public MsflowSocketClient(string url)
        {
            Url = url;
        }

        /// <summary>Drain the main-thread callback queue (call from Update).</summary>
        public void PumpMain()
        {
            while (_mainThreadQueue.TryDequeue(out var action))
            {
                action();
            }
        }

        /// <summary>Connect and wait for the server hello (capabilities).</summary>
        public async Task<MsflowServerCapabilities> ConnectAsync(CancellationToken ct = default)
        {
            ThrowIfDisposed();
            if (IsConnected)
            {
                return await WaitHello(ct);
            }
            Volatile.Write(ref _state, StateConnecting);
            _cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            _helloWaiter = new TaskCompletionSource<MsflowServerCapabilities>(
                TaskCreationOptions.RunContinuationsAsynchronously);
            var socket = new ClientWebSocket();
            _socket = socket;
            try
            {
                await socket.ConnectAsync(new Uri(Url), _cts.Token).ConfigureAwait(true);
            }
            catch
            {
                Volatile.Write(ref _state, StateClosed);
                throw;
            }
            Volatile.Write(ref _state, StateConnected);
            _sessionTask = Task.Run(() => SessionLoopAsync(socket, _cts.Token));
            // protocol v1: the server only answers hello with its capabilities
            await SendAsync(
                new JObject { ["type"] = "hello", ["protocol"] = 1 }, ct).ConfigureAwait(true);
            var capabilities = await WaitHello(ct).ConfigureAwait(true);
            EnqueueMain(() => Connected?.Invoke());
            return capabilities;
        }

        private async Task<MsflowServerCapabilities> WaitHello(CancellationToken ct)
        {
            var waiter = _helloWaiter;
            var timeout = Task.Delay(10000, ct);
            var finished = await Task.WhenAny(waiter.Task, timeout).ConfigureAwait(true);
            if (finished != waiter.Task)
            {
                throw new TimeoutException("server did not answer hello within 10s");
            }
            return waiter.Task.Result;
        }

        /// <summary>Queue a text-only or constrained generation; returns the job id.</summary>
        public async Task<string> GenerateAsync(
            MsflowGenerateRequest request, CancellationToken ct = default)
        {
            if (request == null)
            {
                throw new ArgumentNullException(nameof(request));
            }
            if (string.IsNullOrWhiteSpace(request.Prompt))
            {
                throw new ArgumentException("prompt is required", nameof(request));
            }
            ThrowIfDisposed();
            if (!IsConnected)
            {
                throw new InvalidOperationException("not connected; call ConnectAsync first");
            }
            var jobId = Guid.NewGuid().ToString("N");
            lock (_gate)
            {
                _pending[jobId] = request;
            }
            await SendAsync(BuildGenerateMessage(request, jobId), ct).ConfigureAwait(true);
            return jobId;
        }

        /// <summary>Re-request a job's result (after reconnect).</summary>
        public Task FetchAsync(string jobId, CancellationToken ct = default)
        {
            return SendAsync(new JObject { ["type"] = "fetch", ["job_id"] = jobId }, ct);
        }

        /// <summary>Best-effort cancel of a queued/running job.</summary>
        public Task CancelAsync(string jobId, CancellationToken ct = default)
        {
            return SendAsync(new JObject { ["type"] = "cancel", ["job_id"] = jobId }, ct);
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            Volatile.Write(ref _state, StateClosed);
            try
            {
                _cts?.Cancel();
            }
            catch (ObjectDisposedException)
            {
            }
            try
            {
                _socket?.Dispose();
            }
            catch (ObjectDisposedException)
            {
            }
            EnqueueMain(() => Disconnected?.Invoke("disposed"));
        }

        // -- protocol ---------------------------------------------------------

        internal JObject BuildGenerateMessage(MsflowGenerateRequest request, string jobId)
        {
            var message = new JObject
            {
                ["type"] = "generate",
                ["job_id"] = jobId,
                ["prompt"] = request.Prompt,
            };
            if (request.NFrames.HasValue)
            {
                message["n_frames"] = request.NFrames.Value;
            }
            else
            {
                message["duration_sec"] = request.DurationSec;
            }
            if (request.Seed.HasValue)
            {
                message["seed"] = request.Seed.Value;
            }
            if (request.Cfg.HasValue)
            {
                message["cfg"] = request.Cfg.Value;
            }
            if (request.NumSteps.HasValue)
            {
                message["num_steps"] = request.NumSteps.Value;
            }
            if (request.Controls != null && request.Controls.Count > 0)
            {
                var controls = new JArray();
                foreach (var control in request.Controls)
                {
                    controls.Add(control.ToJObject());
                }
                message["controls"] = controls;
            }
            return message;
        }

        private async Task SendAsync(JObject message, CancellationToken ct)
        {
            var bytes = Encoding.UTF8.GetBytes(message.ToString(Newtonsoft.Json.Formatting.None));
            var socket = _socket;
            if (socket == null)
            {
                throw new InvalidOperationException("not connected");
            }
            await _sendLock.WaitAsync(ct).ConfigureAwait(true);
            try
            {
                await socket.SendAsync(
                    new ArraySegment<byte>(bytes),
                    WebSocketMessageType.Text,
                    true,
                    ct).ConfigureAwait(true);
            }
            finally
            {
                _sendLock.Release();
            }
        }

        private async Task SessionLoopAsync(ClientWebSocket socket, CancellationToken ct)
        {
            var buffer = new byte[64 * 1024];
            var builder = new StringBuilder();
            try
            {
                while (!ct.IsCancellationRequested && !_disposed)
                {
                    builder.Clear();
                    WebSocketReceiveResult result;
                    do
                    {
                        result = await socket.ReceiveAsync(
                            new ArraySegment<byte>(buffer), ct).ConfigureAwait(false);
                        builder.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
                    }
                    while (!result.EndOfMessage && !ct.IsCancellationRequested);
                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        break;
                    }
                    var json = builder.ToString();
                    EnqueueMain(() => HandleMessage(json));
                }
            }
            catch (OperationCanceledException)
            {
            }
            catch (Exception exc)
            {
                var message = exc.Message;
                EnqueueMain(() => Error?.Invoke("", $"connection lost: {message}"));
            }
            finally
            {
                Volatile.Write(ref _state, StateClosed);
                var reason = _disposed ? "disposed" : "closed";
                EnqueueMain(() => Disconnected?.Invoke(reason));
                EnqueueMain(() => MaybeReconnect());
            }
        }

        private void MaybeReconnect()
        {
            if (_disposed || !AutoReconnect)
            {
                return;
            }
            Task.Run(async () =>
            {
                while (!_disposed && Volatile.Read(ref _state) != StateConnected)
                {
                    await Task.Delay(TimeSpan.FromSeconds(ReconnectDelaySeconds)).ConfigureAwait(false);
                    if (_disposed)
                    {
                        return;
                    }
                    try
                    {
                        await ConnectAsync().ConfigureAwait(false);
                        // re-fetch unfinished jobs so a drop does not lose work
                        List<string> unfinished;
                        lock (_gate)
                        {
                            unfinished = new List<string>(_pending.Keys);
                        }
                        foreach (var jobId in unfinished)
                        {
                            await FetchAsync(jobId).ConfigureAwait(false);
                        }
                    }
                    catch
                    {
                        // keep retrying until disposed or reconnected
                    }
                }
            });
        }

        private void HandleMessage(string json)
        {
            JObject message;
            try
            {
                message = JObject.Parse(json);
            }
            catch (Exception exc)
            {
                Error?.Invoke("", $"bad server message: {exc.Message}");
                return;
            }
            var type = message.Value<string>("type") ?? "";
            var jobId = message.Value<string>("job_id") ?? "";
            switch (type)
            {
                case "hello":
                    var capabilities = new MsflowServerCapabilities
                    {
                        Protocol = message.Value<int?>("protocol") ?? 0,
                        Fps = message.Value<float?>("fps") ?? 20f,
                        MaxLength = message.Value<int?>("max_length"),
                        Model = message.Value<string>("model") ?? "",
                        Ckpt = message.Value<string>("ckpt") ?? "",
                    };
                    _helloWaiter?.TrySetResult(capabilities);
                    Hello?.Invoke(capabilities);
                    break;
                case "accepted":
                    // informational; the job id is already tracked client-side
                    break;
                case "progress":
                    Progress?.Invoke(
                        jobId,
                        message.Value<float?>("fraction") ?? 0f,
                        message.Value<string>("message") ?? "");
                    break;
                case "result":
                    var raw = message["motion"] as JObject;
                    if (raw == null)
                    {
                        Error?.Invoke(jobId, "result message has no motion document");
                        break;
                    }
                    MsflowMotion motion;
                    try
                    {
                        motion = MsflowMotion.FromJObject(raw);
                    }
                    catch (MsflowMotionException exc)
                    {
                        Error?.Invoke(jobId, $"invalid motion document: {exc.Message}");
                        break;
                    }
                    ForgetPending(jobId);
                    Result?.Invoke(jobId, motion, raw.ToString(Newtonsoft.Json.Formatting.None));
                    break;
                case "error":
                    ForgetPending(jobId);
                    Error?.Invoke(jobId, message.Value<string>("message") ?? "unknown error");
                    break;
                default:
                    Error?.Invoke(jobId, $"unknown server message type '{type}'");
                    break;
            }
        }

        private void ForgetPending(string jobId)
        {
            lock (_gate)
            {
                _pending.Remove(jobId);
            }
        }

        private void EnqueueMain(Action action)
        {
            _mainThreadQueue.Enqueue(action);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(MsflowSocketClient));
            }
        }
    }
}
