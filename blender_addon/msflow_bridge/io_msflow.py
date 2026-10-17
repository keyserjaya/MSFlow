"""Parse `msflow-motion` JSON documents (schema v1) for the Blender bridge.

Deliberately pure python+numpy (no bpy) so the tests can run it both inside
Blender (bundled numpy) and outside from a plain interpreter.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_NAME = "msflow-motion"
FORMAT_VERSION = 1
NUM_JOINTS = 22


class MsflowFormatError(ValueError):
    pass


@dataclass
class MotionData:
    fps: float
    positions: np.ndarray  # [T, 22, 3] MSFlow world space (Y-up, metres)
    joint_names: list[str]
    parents: list[int]
    n_frames: int
    duration_sec: float
    controls: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    source: str = ""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MsflowFormatError(message)


def default_action_label(meta: dict[str, Any]) -> str:
    """Stable Action name, e.g. `MSFlow_kick_20260101-120000_s00`."""
    import re

    run = str(meta.get("source_run") or "motion")
    run = run.replace("/", "_")
    sample = meta.get("sample_index")
    suffix = f"_s{int(sample):02d}" if isinstance(sample, int) else ""
    label = f"MSFlow_{run}{suffix}"
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", label) or "msflow_motion"


def parse_motion_dict(raw: Any, source: str = "") -> MotionData:
    _require(isinstance(raw, dict), "document is not a JSON object")
    _require(
        raw.get("format") == FORMAT_NAME,
        f"format must be '{FORMAT_NAME}', got {raw.get('format')!r}",
    )
    version = raw.get("version")
    _require(
        isinstance(version, int) and version == FORMAT_VERSION,
        f"only version {FORMAT_VERSION} is supported, got {version!r}",
    )
    fps = float(raw.get("fps", 0))
    _require(fps > 0, f"fps must be positive, got {fps}")
    positions = np.asarray(raw.get("positions"), dtype=np.float32)
    _require(
        positions.ndim == 3 and positions.shape[1:] == (NUM_JOINTS, 3),
        f"positions must be shaped (T, {NUM_JOINTS}, 3), got {positions.shape}",
    )
    n_frames = int(positions.shape[0])
    _require(n_frames > 0, "positions must contain at least one frame")

    names = [str(name) for name in (raw.get("joints") or [])]
    _require(
        len(names) == NUM_JOINTS,
        f"joints must list {NUM_JOINTS} names, got {len(names)}",
    )
    parents_raw = raw.get("parents") or []
    _require(
        len(parents_raw) == NUM_JOINTS,
        f"parents must list {NUM_JOINTS} entries, got {len(parents_raw)}",
    )
    parents = [int(value) for value in parents_raw]
    roots = [joint for joint, parent in enumerate(parents) if parent == -1]
    _require(
        roots == [0],
        f"exactly joint 0 may be parentless (roots={roots})",
    )
    for joint, parent in enumerate(parents):
        _require(parent < joint, f"parent[{joint}]={parent} breaks tree order")

    controls = list(raw.get("controls") or [])
    meta = dict(raw.get("meta") or {})
    duration = float(meta.get("duration_sec", raw.get("duration_sec", n_frames / fps)))
    return MotionData(
        fps=fps,
        positions=positions,
        joint_names=names,
        parents=parents,
        n_frames=n_frames,
        duration_sec=duration,
        controls=controls,
        meta=meta,
        source=source,
    )


def load_motion(path: str | Path) -> MotionData:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MsflowFormatError(f"{path}: invalid JSON ({exc})") from exc
    return parse_motion_dict(raw, source=str(path))


def find_motion_files(out_root: str | Path) -> list[Path]:
    """Every sample_*.msflow.json under the run-output root, newest first."""
    root = Path(out_root)
    if not root.is_dir():
        return []
    files = sorted(root.rglob("sample_*.msflow.json"), key=lambda p: p.stat().st_mtime)
    files.reverse()
    return files


def to_blender_space(world: np.ndarray, flip_facing: bool = False) -> np.ndarray:
    """MSFlow world (Y-up, +Z forward) -> Blender (Z-up, metres): (x, -z, y) —
    a proper rotation, so left/right chirality is preserved. `flip_facing`
    adds a 180° turn about Z if the rig should face the other way."""
    x = world[..., 0]
    up = world[..., 1]
    fwd = world[..., 2]
    blender = np.stack([x, -fwd, up], axis=-1)
    if flip_facing:
        blender[..., 0] = -blender[..., 0]
        blender[..., 1] = -blender[..., 1]
    return blender
