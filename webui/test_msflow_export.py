"""Headless tests for the webui Blender-export path (numpy only, no GPU).

Run from the repo root:
    uv run --group ui python webui/test_msflow_export.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webui.blender_bridge import push_motion  # noqa: E402
from webui.msflow_export import (  # noqa: E402
    FORMAT_NAME,
    FORMAT_VERSION,
    NUM_JOINTS,
    build_motion_dict,
    export_run,
    export_sample,
    joint_parents,
)
from webui.skeleton import JOINT_CHAIN, JOINT_NAMES  # noqa: E402


def synthetic_run(tmp: Path, n_samples: int = 2, t: int = 21) -> Path:
    """A minimal fake run dir: samples_world.npy + spec.json snapshot."""
    rng = np.random.default_rng(3407)
    samples = rng.random((n_samples, t, NUM_JOINTS, 3), dtype=np.float32)
    samples[..., 1] = np.clip(samples[..., 1], 0.0, 2.0)  # plausible heights
    np.save(tmp / "samples_world.npy", samples)
    (tmp / "spec.json").write_text(
        json.dumps(
            {
                "prompt": "a person waves",
                "n_frames": t,
                "controls": [{"joints": [21], "frame": 0, "axes": "x,y,z"}],
            }
        ),
        encoding="utf-8",
    )
    return tmp


def check_dict_schema() -> None:
    parents = joint_parents()
    chain_joints = {j for pair in JOINT_CHAIN for j in pair}
    for joint in range(NUM_JOINTS):
        if joint == 0:
            assert parents[joint] == -1
        else:
            assert parents[joint] in chain_joints, joint
    pose = np.zeros((5, NUM_JOINTS, 3), dtype=np.float32)
    doc = build_motion_dict(pose)
    assert doc["format"] == FORMAT_NAME and doc["version"] == FORMAT_VERSION
    assert doc["fps"] == 20 and doc["n_frames"] == 5
    assert abs(doc["duration_sec"] - 0.25) < 1e-9
    assert doc["joints"] == [JOINT_NAMES[i] for i in range(NUM_JOINTS)]
    assert doc["parents"] == parents
    assert len(doc["positions"]) == 5
    try:
        build_motion_dict(np.zeros((4, 3, 3), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("bad shape must be rejected")
    try:
        build_motion_dict(np.zeros((0, NUM_JOINTS, 3), dtype=np.float32))
        build_motion_dict(np.zeros((5, NUM_JOINTS, 3), dtype=np.float32), fps=0)
    except ValueError:
        pass
    else:
        raise AssertionError("empty frames / zero fps must be rejected")
    print("msflow dict ok (schema, parents, shape/fps guards)")


def check_export_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        run = synthetic_run(Path(tmp_name))
        paths = export_run(run)
        assert [p.name for p in paths] == ["sample_00.msflow.json", "sample_01.msflow.json"]
        doc = json.loads(paths[0].read_text(encoding="utf-8"))
        samples = np.load(run / "samples_world.npy")
        rebuilt = np.asarray(doc["positions"], dtype=np.float32)
        assert np.allclose(rebuilt, np.round(samples[0], 6)), "positions parity"
        assert doc["meta"]["prompt"] == "a person waves"
        assert doc["meta"]["sample_index"] == 0
        assert doc["controls"] == [{"joints": [21], "frame": 0, "axes": "x,y,z"}]
        try:
            export_sample(run, 5)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-range sample must be rejected")
        try:
            export_sample(Path(tmp_name) / "nope")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing run dir must be reported")
        single = export_sample(run, 1, out_path=Path(tmp_name) / "other.json")
        assert single.name == "other.json"
    print("msflow export ok (files, parity, meta passthrough, error paths)")


def check_bridge_errors() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        path = Path(tmp_name) / "sample_00.msflow.json"
        path.write_text("{}", encoding="utf-8")
        ok, message = push_motion(path, host="127.0.0.1", port=1, timeout=0.5)
        assert not ok and message, message
        ok, message = push_motion(Path(tmp_name) / "missing.json")
        assert not ok and "not found" in message, message
    from webui.blender_bridge import windows_visible
    assert windows_visible(Path("/mnt/g/AIProjects/MSFlow/run/x.json")) == (
        "G:\\AIProjects\\MSFlow\\run\\x.json"
    )
    assert windows_visible(Path("C:\\Windows\\x.json")) == "C:\\Windows\\x.json"
    print("blender push ok (graceful failure when the receiver is off)")


if __name__ == "__main__":
    check_dict_schema()
    check_export_roundtrip()
    check_bridge_errors()
    print("msflow export tests OK")
