"""Unity runtime WebSocket bridge.

One connection = one Unity client session. Messages (JSON, protocol v1):

    client -> server: hello | generate | fetch | cancel
    server -> client: hello | accepted | progress | result | error

``generate`` may carry sparse-control ``controls`` (the webui spec-loader
schema; constraint generation via ``webui.runner.Runner``) or arrive without
them (plain text-to-motion via ``webui.runtime_generate``). Results are
``msflow-motion`` v1 documents — the same schema the Blender bridge imports —
and are also written under ``--out-root`` so the Blender folder-watch sees
Unity-triggered runs.

Torch-dependent imports are lazy (inside ``_ModelCore``) so the module can be
imported and protocol-tested without the research stack; inject
``text_generator``/``runner`` callables to stub generation.

    uv run --group server python webui/runtime_server.py --port 8766
"""
from __future__ import annotations

import argparse
import asyncio
import json
import queue
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webui.msflow_export import build_motion_dict  # noqa: E402
from webui.skeleton import JOINT_NAMES  # noqa: E402

PROTOCOL_VERSION = 1
SOURCE_FPS = 20.0
JOB_CACHE_SIZE = 8
JOB_CACHE_TTL_SEC = 300.0
MAX_QUEUE_DEPTH = 16
TEXT_SOURCE_RUN = "text"  # artifacts: <out_root>/text/<ts>/...

ProgressCb = Callable[[float, str], None]
SENTINEL: dict[str, Any] = {}


class JobCancelled(Exception):
    """Raised inside a progress callback when the client cancelled the job."""


@dataclass
class ServerConfig:
    port: int = 8766
    bind: str = "127.0.0.1"
    device: str = "cuda:0"
    model_name: str = "MMDiT_xyz_pretrained"
    ckpt: str = "latest"
    dataset_name: str = "t2m"
    dataset_dir: str = "./datasets"
    checkpoints_dir: str = "./checkpoints"
    out_root: str = "generations/runtime"
    num_steps: int = 300
    cfg: float = 3.0
    save_mp4: bool = True


@dataclass
class GenerateRequest:
    prompt: str
    n_frames: int
    seed: int = 3407
    cfg: float | None = None
    num_steps: int | None = None
    controls: list[dict[str, Any]] | None = None  # None/[] => text-only mode


@dataclass
class RunSettingsMirror:
    """Duck-typed ``webui.runner.RunSettings`` (field-for-field) so the
    constraint path can build settings without importing torch at module
    scope; Runner.run only reads attributes. Keep in sync with the original."""

    name: str = "MMDiT_xyz_pretrained"
    ckpt: str = "latest"
    device: str = "cuda:0"
    dataset_name: str = "t2m"
    dataset_dir: str = "./datasets"
    checkpoints_dir: str = "./checkpoints"
    num_samples: int = 1
    num_steps: int = 300
    cfg: float = 2.0
    w_kin: float = 1.0
    seed: int = 3407
    use_metric_R: bool = True
    use_augmented_obs: bool = True
    use_noise_mixing: bool = True
    noise_mix_strength: float = 1.0
    save_mp4: bool = False
    save_gif: bool = False
    out_root: str = "generations/runtime"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    id: str
    request: GenerateRequest
    owner: "_Connection | None"
    status: str = "queued"  # queued|running|done|error|cancelled
    fraction: float = 0.0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    out_dir: str = ""
    cancel_event: threading.Event = field(default_factory=threading.Event)
    created: float = field(default_factory=time.time)


class JobStore:
    """Job index + result cache: last ``JOB_CACHE_SIZE`` jobs, TTL-bounded."""

    def __init__(self) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, job: Job) -> None:
        with self._lock:
            self._jobs.pop(job.id, None)
            self._jobs[job.id] = job
            self._evict_locked()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.created + JOB_CACHE_TTL_SEC < time.time():
            with self._lock:
                if self._jobs.get(job_id) is job:
                    self._jobs.pop(job_id, None)
            return None
        return job

    def queued_ids(self) -> list[str]:
        with self._lock:
            return [job.id for job in self._jobs.values() if job.status == "queued"]

    def _evict_locked(self) -> None:
        now = time.time()
        while len(self._jobs) > 1:
            oldest, job = next(iter(self._jobs.items()))
            if len(self._jobs) <= JOB_CACHE_SIZE and job.created + JOB_CACHE_TTL_SEC >= now:
                break
            self._jobs.pop(oldest, None)


class _Connection:
    """Outgoing message pump: producers (any thread) enqueue; one task sends."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.alive = True
        self._loop = loop
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    def post(self, message: dict[str, Any]) -> None:
        """Thread-safe enqueue for the sender task (ignored once closed)."""
        if not self.alive:
            return
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, dict(message))
        except RuntimeError:  # loop already torn down during shutdown
            self.alive = False

    async def finish(self, send_task: Any) -> None:
        self.alive = False
        self._queue.put_nowait(SENTINEL)
        try:
            await send_task
        except Exception:
            pass


class RuntimeServer:
    """Job registry + worker thread; ``handle`` is the websockets entry point."""

    def __init__(
        self,
        config: ServerConfig,
        *,
        text_generator: Any | None = None,
        runner: Any | None = None,
    ) -> None:
        self.config = config
        self.jobs = JobStore()
        self._queue: queue.Queue[str] = queue.Queue()
        self._text_generator = text_generator
        self._runner_impl = runner
        self._worker: threading.Thread | None = None
        self._worker_started = False

    # -- model core (lazy; injectable for tests) ---------------------------

    @property
    def text_generator(self) -> Any:
        return self._text_generator if self._text_generator is not None else self._ensure_core().text

    @property
    def runner(self) -> Any:
        return self._runner_impl if self._runner_impl is not None else self._ensure_core().runner

    def _ensure_core(self) -> "_ModelCore":
        """Populate both generator and runner from one real model core (lazy
        torch import) unless pieces were injected (tests)."""
        self._core = self._core or _ModelCore(self.config)
        if self._text_generator is None:
            self._text_generator = self._core.text
        if self._runner_impl is None:
            self._runner_impl = self._core.runner
        return self._core

    def start_worker(self) -> None:
        if self._worker_started:
            return
        self._worker_started = True
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="msflow-runtime"
        )
        self._worker.start()

    # -- websocket plumbing --------------------------------------------------

    async def handle(self, websocket: Any) -> None:
        connection = _Connection(asyncio.get_running_loop())
        sender = asyncio.create_task(
            self._sender(websocket, connection), name="msflow-sender"
        )
        try:
            async for raw in websocket:
                reply = self.consume_message(connection, raw)
                if reply is not None:
                    connection.post(reply)
        except Exception:
            pass  # normal close paths (ConnectionClosed) surface here; reply sent already
        finally:
            await connection.finish(sender)

    async def _sender(self, websocket: Any, connection: _Connection) -> None:
        while True:
            message = await connection._queue.get()
            if message == SENTINEL:
                return
            await websocket.send(json.dumps(message))

    # -- inbound messages ---------------------------------------------------

    def consume_message(self, connection: _Connection, raw: Any) -> dict[str, Any]:
        """Parse one inbound message; returns the immediate reply message."""
        try:
            payload = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            return _error("", f"bad JSON body: {exc}")
        if not isinstance(payload, dict):
            return _error("", "payload must be a JSON object")
        kind = str(payload.get("type", ""))
        if kind == "hello":
            return self._hello_reply()
        if kind == "generate":
            return self._start_job(connection, payload)
        if kind == "fetch":
            return self._fetch(payload)
        if kind == "cancel":
            return self._cancel(payload)
        return _error("", f"unknown message type {kind!r}")

    def _hello_reply(self) -> dict[str, Any]:
        max_length = None
        if self._text_generator is not None:
            try:
                max_length = self._text_generator.max_length()
            except Exception:
                max_length = None
        return {
            "type": "hello",
            "protocol": PROTOCOL_VERSION,
            "fps": SOURCE_FPS,
            "max_length": max_length,
            "model": self.config.model_name,
            "ckpt": self.config.ckpt,
        }

    def _start_job(self, connection: _Connection, payload: dict) -> dict[str, Any]:
        request, failure = _parse_generate_request(payload)
        if failure is not None or request is None:
            return failure or _error("", "invalid generate request")
        limit = self._cached_max_length()
        if limit and request.n_frames > limit:
            return _error(
                str(payload.get("job_id") or ""),
                f"n_frames {request.n_frames} exceeds model max_length {limit}",
            )
        queued_ahead = len(self.jobs.queued_ids())
        if queued_ahead >= MAX_QUEUE_DEPTH:
            return _error("", f"queue full ({queued_ahead} waiting); retry later")
        job = Job(
            id=str(payload.get("job_id") or uuid.uuid4().hex),
            request=request,
            owner=connection,
        )
        self.jobs.put(job)
        # enqueue before starting the worker so `accepted` always arrives first
        connection.post(
            {"type": "accepted", "job_id": job.id, "queue_position": queued_ahead}
        )
        self._queue.put(job.id)
        self.start_worker()
        return None

    def _fetch(self, payload: dict) -> dict[str, Any]:
        job_id = str(payload.get("job_id") or "")
        job = self.jobs.get(job_id) if job_id else None
        if job is None:
            return _error(job_id, "unknown job_id")
        if job.status == "done" and job.result is not None:
            return _result(job_id, job.result)
        if job.status == "error":
            return _error(job_id, job.error or "job failed")
        if job.status in ("queued", "running"):
            return {
                "type": "progress",
                "job_id": job_id,
                "fraction": job.fraction,
                "message": f"{job.status}: {job.message}".strip(),
            }
        return _error(job_id, "job was cancelled")

    def _cancel(self, payload: dict) -> dict[str, Any]:
        job_id = str(payload.get("job_id") or "")
        job = self.jobs.get(job_id) if job_id else None
        if job is None:
            return _error(job_id, "unknown job_id")
        if job.status in ("queued", "running"):
            job.cancel_event.set()
            return {"type": "accepted", "job_id": job_id, "cancelled": True}
        return {"type": "accepted", "job_id": job_id, "cancelled": False}

    # -- worker thread -------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.jobs.get(job_id)
            if job is not None:
                self._run_job(job)

    def _run_job(self, job: Job) -> None:
        connection = job.owner
        if job.cancel_event.is_set():
            _finish_cancelled(job)
            return
        request = job.request
        job.status = "running"
        job.fraction = 0.01
        job.message = "starting"
        if connection is not None:
            connection.post(
                _progress(job.id, job.fraction, f"starting: {request.prompt[:80]}")
            )
        try:
            if request.controls is not None:
                out_dir = self._generate_constrained(job)
            else:
                out_dir = self._generate_text(job)
        except Exception as exc:
            if isinstance(exc, JobCancelled):
                _finish_cancelled(job)
                self.jobs.put(job)
                return
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            self.jobs.put(job)
            if connection is not None:
                connection.post(_error(job.id, job.error))
            return
        job.status = "done"
        job.fraction = 1.0
        job.message = "done"
        self.jobs.put(job)
        if connection is not None:
            connection.post(_result(job.id, job.result or {}))

    # -- generation backends ---------------------------------------------------

    def _generate_constrained(self, job: Job) -> str:
        config = self.config
        request = job.request
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        settings = RunSettingsMirror(
            name=config.model_name,
            ckpt=config.ckpt,
            device=config.device,
            dataset_name=config.dataset_name,
            dataset_dir=config.dataset_dir,
            checkpoints_dir=config.checkpoints_dir,
            num_samples=1,
            num_steps=request.num_steps if request.num_steps else config.num_steps,
            cfg=request.cfg if request.cfg is not None else config.cfg,
            w_kin=1.0,
            seed=request.seed,
            save_mp4=config.save_mp4,
            save_gif=False,
            out_root=config.out_root,
        )
        spec = {
            "spec_name": f"unity_{job.id[:8]}_{timestamp}",
            "prompt": request.prompt,
            "n_frames": request.n_frames,
            "controls": _with_stage_defaults(request.controls, request.n_frames),
            "joint_names": {str(k): name for k, name in JOINT_NAMES.items()},
        }
        result = self.runner.run(spec, settings, progress_cb=_progress_cb(job))
        out_dir = Path(result.out_dir)
        sample_path = out_dir / "sample_00.msflow.json"
        if sample_path.is_file():
            job.result = json.loads(sample_path.read_text(encoding="utf-8"))
        elif result.motion is not None:
            job.result = build_motion_dict(
                np.asarray(result.motion),
                controls=list(spec["controls"] or []),
                meta={
                    "prompt": request.prompt,
                    "source_run": f"{out_dir.parent.name}/{out_dir.name}",
                    "sample_index": 0,
                    "created": timestamp,
                    "world_source": "RunResult.motion",
                },
            )
        else:
            raise RuntimeError(
                f"generation completed but no export was produced in {out_dir}"
            )
        job.out_dir = str(out_dir)
        return job.out_dir

    def _generate_text(self, job: Job) -> str:
        request = job.request
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        result = self.text_generator.generate(
            request.prompt,
            n_frames=request.n_frames,
            seed=request.seed,
            cfg=request.cfg if request.cfg is not None else self.config.cfg,
            progress_cb=_progress_cb(job),
        )
        doc = build_motion_dict(
            result.positions,
            controls=[],
            meta={
                "prompt": request.prompt,
                "source_run": f"{TEXT_SOURCE_RUN}/text/{timestamp}",
                "sample_index": 0,
                "created": timestamp,
                "seed": request.seed,
            },
        )
        out_dir = Path(self.config.out_root) / "text" / timestamp
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "sample_00.msflow.json").write_text(
                json.dumps(doc), encoding="utf-8"
            )
            if self.config.save_mp4:
                _render_motion_mp4(
                    out_dir / "sample_00.mp4", result.positions, request.prompt
                )
        except OSError as exc:
            print(f"[runtime] artifact write failed: {exc}", flush=True)
        job.result = doc
        job.out_dir = str(out_dir)
        return job.out_dir

    def _cached_max_length(self) -> int | None:
        generator = self._text_generator
        if generator is None:
            return None
        try:
            return generator.max_length()
        except Exception:
            return None

    _core: Any = None  # lazily built real model core (tests inject fakes instead)


def _with_stage_defaults(
    controls: list[dict[str, Any]], n_frames: int
) -> list[dict[str, Any]]:
    """The constraint sampler's hint-color pass expects a `stage` per control
    (the webUI editor always writes one; the WS protocol treats it as
    optional) — default to the webUI's early/middle/late thirds."""
    normalized = []
    span = max(n_frames - 1, 1)
    for entry in controls:
        item = dict(entry)
        if not item.get("stage"):
            frame = item.get("frame")
            if frame is None and item.get("frame_fraction") is not None:
                frame = float(item["frame_fraction"]) * span
            if frame is None:
                item["stage"] = "middle"
            else:
                fraction = float(frame) / span
                item["stage"] = (
                    "early" if fraction < 1.0 / 3.0
                    else "middle" if fraction < 2.0 / 3.0
                    else "late"
                )
        normalized.append(item)
    return normalized


def _render_motion_mp4(path: Path, positions: Any, text: str) -> None:
    """Best-effort mp4 (matplotlib); never fails a finished generation."""
    try:
        from utils.plot_script import plot_3d_motion

        plot_3d_motion(str(path), positions, text, fps=20, radius=4)
    except Exception as exc:  # pragma: no cover - rendering is optional
        print(f"[runtime] mp4 render failed: {exc}", flush=True)


def _progress_cb(job: Job) -> ProgressCb:
    """Progress callback bound to one job: caches state, streams to owner,
    raises JobCancelled when the client asked to cancel this job."""
    def cb(fraction: float, message: str) -> None:
        if job.cancel_event.is_set():
            raise JobCancelled()
        job.fraction = fraction
        job.message = message
        if job.owner is not None:
            job.owner.post(_progress(job.id, fraction, message))

    return cb


def _finish_cancelled(job: Job) -> None:
    job.status = "cancelled"
    job.message = "cancelled"
    if job.owner is not None:
        job.owner.post(_error(job.id, "cancelled"))


def _error(job_id: str, message: str) -> dict[str, Any]:
    return {"type": "error", "job_id": job_id, "message": message}


def _progress(job_id: str, fraction: float, message: str) -> dict[str, Any]:
    return {
        "type": "progress",
        "job_id": job_id,
        "fraction": round(float(fraction), 4),
        "message": message,
    }


def _result(job_id: str, motion: dict[str, Any]) -> dict[str, Any]:
    return {"type": "result", "job_id": job_id, "motion": motion}


def _parse_generate_request(payload: dict) -> tuple[GenerateRequest | None, dict[str, Any] | None]:
    """Validate a generate payload; returns (request, None) or (None, error)."""
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        return None, _error(str(payload.get("job_id") or ""), "prompt is required")

    n_frames: int
    if payload.get("n_frames") is not None:
        try:
            n_frames = int(payload["n_frames"])
        except (TypeError, ValueError):
            return None, _error("", "n_frames must be an integer")
    else:
        duration = payload.get("duration_sec")
        if duration is None:
            duration = 6.0
        try:
            n_frames = int(round(float(duration) * SOURCE_FPS))
        except (TypeError, ValueError):
            return None, _error("", "duration_sec must be a number")
    if not 1 <= n_frames <= 4096:
        return None, _error("", f"n_frames must be in [1, 4096], got {n_frames}")

    seed = 3407
    if payload.get("seed") is not None:
        try:
            seed = int(payload["seed"])
        except (TypeError, ValueError):
            return None, _error("", "seed must be an integer")

    cfg = None
    if payload.get("cfg") is not None:
        try:
            cfg = float(payload["cfg"])
        except (TypeError, ValueError):
            return None, _error("", "cfg must be a number")

    num_steps = None
    if payload.get("num_steps") is not None:
        try:
            num_steps = int(payload["num_steps"])
        except (TypeError, ValueError):
            return None, _error("", "num_steps must be an integer")

    controls = payload.get("controls")
    if controls is not None and not isinstance(controls, list):
        return None, _error("", "controls must be a list")

    request = GenerateRequest(
        prompt=prompt,
        n_frames=n_frames,
        seed=seed,
        cfg=cfg,
        num_steps=num_steps,
        controls=controls if controls else None,
    )
    return request, None


class _ModelCore:
    """The real torch-backed model core (imported lazily).

    Keeps ONE ``ModelCache`` so text-only and constraint jobs share VRAM; the
    constraint ``Runner`` is subclassed to load through the same cache.
    """

    def __init__(self, config: ServerConfig) -> None:
        from webui.runner import Runner
        from webui.runtime_generate import (
            ModelCache,
            ModelSettings,
            TextMotionGenerator,
        )

        self.config = config
        self.settings = ModelSettings(
            name=config.model_name,
            ckpt=config.ckpt,
            device=config.device,
            dataset_name=config.dataset_name,
            dataset_dir=config.dataset_dir,
            checkpoints_dir=config.checkpoints_dir,
        )
        self.cache = ModelCache()
        self.text = TextMotionGenerator(self.cache, self.settings)

        class CacheBridgedRunner(Runner):
            def __init__(runner_self, core: _ModelCore) -> None:
                super().__init__()
                runner_self._bridge = core

            def _load_model(
                runner_self,
                settings: Any,
                device: Any,
                progress_cb: ProgressCb | None,
            ) -> Any:
                shared = ModelSettings(
                    name=settings.name,
                    ckpt=settings.ckpt,
                    device=runner_self._bridge.config.device,
                    dataset_name=settings.dataset_name,
                    dataset_dir=settings.dataset_dir,
                    checkpoints_dir=settings.checkpoints_dir,
                )
                return runner_self._bridge.cache.get(shared, device)

        self.runner = CacheBridgedRunner(self)


# -- CLI -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MSFlow Unity runtime bridge (WebSocket, protocol v1)."
    )
    parser.add_argument("--port", type=int, default=ServerConfig.port)
    parser.add_argument("--bind", default=ServerConfig.bind)
    parser.add_argument(
        "--device",
        default=ServerConfig.device,
        help="torch device (e.g. cuda:0; use 'cpu' to run on CPU)",
    )
    parser.add_argument("--name", default=ServerConfig.model_name)
    parser.add_argument("--ckpt", default=ServerConfig.ckpt)
    parser.add_argument("--dataset-dir", default=ServerConfig.dataset_dir)
    parser.add_argument("--checkpoints-dir", default=ServerConfig.checkpoints_dir)
    parser.add_argument("--out-root", default=ServerConfig.out_root)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=ServerConfig.num_steps,
        help="default sampling steps for constraint jobs",
    )
    parser.add_argument("--cfg", type=float, default=ServerConfig.cfg)
    parser.add_argument(
        "--no-save-mp4",
        action="store_true",
        help="skip rendering sample_00.mp4 next to each run",
    )
    args = parser.parse_args(argv)

    config = ServerConfig(
        port=args.port,
        bind=args.bind,
        device=args.device,
        model_name=args.name,
        ckpt=args.ckpt,
        dataset_dir=args.dataset_dir,
        checkpoints_dir=args.checkpoints_dir,
        out_root=args.out_root,
        num_steps=args.num_steps,
        cfg=args.cfg,
        save_mp4=not args.no_save_mp4,
    )
    server = RuntimeServer(config)
    serve_forever(server, config)


def serve_forever(server: RuntimeServer, config: ServerConfig) -> None:
    import websockets as websockets_module

    async def serve() -> None:
        async with websockets_module.serve(
            server.handle, config.bind, config.port, ping_interval=20
        ):
            print(
                f"[runtime] MSFlow unity bridge on ws://{config.bind}:{config.port} "
                f"(model {config.model_name}/{config.ckpt})",
                flush=True,
            )
            await asyncio.Future()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
