"""Protocol selftest for the Unity runtime WebSocket bridge — no torch, no GPU.

Stores generation behind injected fakes and exercises the wire protocol the
Unity client implements: hello capabilities, text-only and constrained
generate paths (accepted -> progress -> result), fetch-after-reconnect,
error paths, artifact writes, and result-schema validation against the same
``msflow-motion`` parser the Blender bridge uses.

    uv run --group server python webui/test_runtime_server.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets  # noqa: E402

from webui.msflow_export import build_motion_dict  # noqa: E402
from webui.runtime_server import RuntimeServer, ServerConfig  # noqa: E402

JOINTS = 22
FPS = 20


def synthetic_world(n_frames: int, seed: int) -> np.ndarray:
    """Deterministic [T, 22, 3] world joints standing near the canonical pose."""
    t = np.arange(n_frames, dtype=np.float32)[:, None, None]
    j = np.arange(JOINTS, dtype=np.float32)[None, :, None]
    d = j * 0.01  # spread joints below one another on x
    y = 0.95 - d + 0.02 * np.sin((t + j) * 0.1)
    z = 0.01 * np.cos((t + j) * 0.13)
    world = np.concatenate(
        [
            np.broadcast_to(d, (n_frames, JOINTS, 1)),
            np.broadcast_to(y, (n_frames, JOINTS, 1)),
            np.broadcast_to(z, (n_frames, JOINTS, 1)),
        ],
        axis=2,
    ).astype(np.float32)
    del seed  # kept in the signature so fakes mirror the real generator API
    return world


class FakeTextGenerator:
    max_length_value = 500

    def max_length(self) -> int:
        return self.max_length_value

    def generate(self, prompt, *, n_frames, seed, cfg, progress_cb=None):
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt must be a non-empty string.")
        if n_frames > self.max_length():
            raise ValueError(
                f"n_frames must be <= model max_length ({self.max_length()}), "
                f"got {n_frames}."
            )
        if progress_cb:
            progress_cb(0.02, "loading model …")
        if progress_cb:
            progress_cb(0.6, f"step {n_frames} frames, cfg={cfg}")
        if progress_cb:
            progress_cb(1.0, "done")
        world = synthetic_world(int(n_frames), seed)
        return SimpleNamespace(positions=world, n_frames=int(world.shape[0]))


class FakeRunner:
    """Mimics webui.runner.Runner: writes spec.json, npy + sample_00.msflow.json."""

    def __init__(self) -> None:
        self.specs: list[dict] = []

    def run(self, spec, settings, progress_cb=None):
        self.specs.append(dict(spec))
        if progress_cb:
            progress_cb(0.05, "model ready; sampling …")
            progress_cb(0.5, "step 10/20")
            progress_cb(1.0, "done")
        n_frames = int(spec["n_frames"])
        out_dir = Path(settings.out_root) / str(spec["spec_name"]) / "ts0000"
        out_dir.mkdir(parents=True, exist_ok=True)
        world = synthetic_world(n_frames, int(getattr(settings, "seed", 3407)))
        np.save(out_dir / "samples_world.npy", world[None, ...])
        doc = build_motion_dict(
            world,
            controls=list(spec["controls"] or []),
            meta={"prompt": spec["prompt"], "sample_index": 0},
        )
        (out_dir / "sample_00.msflow.json").write_text(
            json.dumps(doc), encoding="utf-8"
        )
        return SimpleNamespace(out_dir=str(out_dir), motion=world.tolist())


def load_io_msflow():
    """Load the Blender bridge's parser without importing its package chain."""
    path = Path(__file__).resolve().parent.parent / (
        "blender_addon/msflow_bridge/io_msflow.py"
    )
    spec = importlib.util.spec_from_file_location("msflow_io_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["msflow_io_under_test"] = module  # dataclasses needs the lookup
    spec.loader.exec_module(module)
    return module


async def recv_until(ws, job_id, terminals=("result", "error"), timeout=15.0):
    """Collect messages until a terminal message for `job_id` arrives."""
    seen = []
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"timeout waiting for {terminals} (saw {seen!r})")
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
        seen.append(message)
        if message.get("job_id") == job_id and message.get("type") in terminals:
            return seen, message


async def expect_error(ws, job_id, needle):
    seen, message = await recv_until(ws, job_id)
    assert message["type"] == "error", f"expected error, saw {seen!r}"
    assert needle in str(message.get("message", "")), message
    return seen, message


async def run_test(port: int) -> None:
    io_msflow = load_io_msflow()
    text_generator = FakeTextGenerator()
    fake_runner = FakeRunner()
    with tempfile.TemporaryDirectory() as tmp:
        # save_mp4=False: mp4 rendering is best-effort decoration (and slow on
        # cold matplotlib caches); the protocol assertions only need the
        # msflow artifacts.
        config = ServerConfig(
            port=port, out_root=str(Path(tmp) / "generations"), save_mp4=False
        )
        server = RuntimeServer(
            config, text_generator=text_generator, runner=fake_runner
        )
        socket_server = await websockets.serve(
            server.handle, "127.0.0.1", port, ping_interval=None
        )
        bound_port = socket_server.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{bound_port}"
        try:
            async with websockets.connect(url, max_size=None) as ws:
                # hello reports capabilities
                await ws.send(json.dumps({"type": "hello", "protocol": 1}))
                hello = json.loads(await ws.recv())
                assert hello["type"] == "hello" and hello["protocol"] == 1, hello
                assert hello["fps"] == FPS
                assert hello["max_length"] == FakeTextGenerator.max_length_value
                assert hello["model"] == config.model_name

                # unknown message types error out
                await ws.send(json.dumps({"type": "nonexistent"}))
                err = json.loads(await ws.recv())
                assert err["type"] == "error" and "unknown message type" in err["message"]

                # bad JSON errors out
                await ws.send("{not json")
                err = json.loads(await ws.recv())
                assert err["type"] == "error" and "bad JSON" in err["message"]

                # text-only generate: accepted -> progress -> result
                await ws.send(
                    json.dumps(
                        {
                            "type": "generate",
                            "job_id": "job-text-1",
                            "prompt": "person waving hand",
                            "duration_sec": 3.0,
                            "seed": 1234,
                        }
                    )
                )
                accepted = json.loads(await ws.recv())
                assert accepted["type"] == "accepted", accepted
                assert accepted["queue_position"] == 0
                seen, result = await recv_until(ws, "job-text-1")
                kinds = [m["type"] for m in seen]
                assert "progress" in kinds and result["type"] == "result", seen
                motion = result["motion"]
                parsed = io_msflow.parse_motion_dict(motion, source="<selftest>")
                assert parsed.fps == FPS and parsed.n_frames == 60
                expected = synthetic_world(60, 1234)
                assert np.allclose(
                    np.asarray(motion["positions"]), expected, atol=2e-6
                ), "text result positions drifted from the generator output"
                # the artifact on disk equals the streamed document
                artifact = Path(config.out_root) / "text"
                files = sorted(artifact.glob("**/sample_00.msflow.json"))
                assert files, f"text artifact missing under {artifact}"
                on_disk = json.loads(files[0].read_text(encoding="utf-8"))
                assert on_disk == motion, "artifact != streamed result"
                io_msflow.parse_motion_dict(on_disk, source=str(files[0]))

                # fetch redelivers the same result
                await ws.send(json.dumps({"type": "fetch", "job_id": "job-text-1"}))
                again = json.loads(await ws.recv())
                assert again["type"] == "result" and again["motion"] == motion, again

            # fetch after reconnect (new connection, same job id)
            async with websockets.connect(url, max_size=None) as ws:
                await ws.send(json.dumps({"type": "fetch", "job_id": "job-text-1"}))
                again = json.loads(await ws.recv())
                assert again["type"] == "result", again

            async with websockets.connect(url, max_size=None) as ws:
                # constrained generate: the fake runner writes the artifact the
                # real Runner would, and the server streams it back verbatim
                control = {
                    "frame": 20,
                    "joints": [20],
                    "axes": "x,y,z",
                    "values": [[0.5, 1.2, 0.3]],
                }
                await ws.send(
                    json.dumps(
                        {
                            "type": "generate",
                            "job_id": "job-ctrl-1",
                            "prompt": "a person kicks forward",
                            "n_frames": 40,
                            "seed": 77,
                            "controls": [control],
                        }
                    )
                )
                accepted = json.loads(await ws.recv())
                assert accepted["type"] == "accepted", accepted
                seen, result = await recv_until(ws, "job-ctrl-1")
                assert result["type"] == "result", seen
                motion = result["motion"]
                io_msflow.parse_motion_dict(motion, source="<selftest-ctrl>")
                # server normalizes controls: stage defaults to the thirds rule
                expected_control = dict(control)
                expected_control["stage"] = "middle"  # frame 20 of 40
                assert motion["controls"] == [expected_control], "controls not passed through"
                assert len(fake_runner.specs) == 1
                ran_spec = fake_runner.specs[0]
                assert ran_spec["prompt"] == "a person kicks forward"
                assert ran_spec["n_frames"] == 40
                assert ran_spec["controls"][0]["stage"] == "middle"
                # on-disk artifact equals streamed doc
                ctrl_files = sorted(
                    Path(config.out_root).glob(
                        "unity_*/**/sample_00.msflow.json"
                    )
                )
                assert ctrl_files, f"constraint artifact missing under {config.out_root}"
                on_disk = json.loads(ctrl_files[0].read_text(encoding="utf-8"))
                assert on_disk == motion

                # protocol errors: empty prompt, oversize length, fetch unknown
                await ws.send(
                    json.dumps({"type": "generate", "job_id": "e1", "prompt": "  "})
                )
                await expect_error(ws, "e1", "prompt is required")
                await ws.send(
                    json.dumps(
                        {
                            "type": "generate",
                            "job_id": "e2",
                            "prompt": "hi",
                            "n_frames": 600,
                        }
                    )
                )
                await expect_error(ws, "e2", "exceeds model max_length")
                await ws.send(json.dumps({"type": "fetch", "job_id": "missing"}))
                await expect_error(ws, "missing", "unknown job_id")
                await ws.send(json.dumps({"type": "cancel", "job_id": "missing"}))
                await expect_error(ws, "missing", "unknown job_id")

                # duration_sec must be a number (parse failures reply with "")
                await ws.send(
                    json.dumps(
                        {
                            "type": "generate",
                            "job_id": "e3",
                            "prompt": "x",
                            "duration_sec": "soon",
                        }
                    )
                )
                await expect_error(ws, "", "duration_sec must be a number")
        finally:
            socket_server.close()
            await socket_server.wait_closed()
    print("runtime_server selftest OK")


def main() -> None:
    asyncio.run(run_test(port=0))


if __name__ == "__main__":
    main()
