"""Export MSFlow run outputs to `msflow-motion` JSON for the Blender bridge.

One JSON per sample lands next to the run's npy/mp4 files (shared between
WSL2 and Windows), so the Blender add-on can import it over the file system
or receive its path via the localhost push. Pure numpy on purpose: loadable
outside the research env (and inside Blender) for tests.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from webui.skeleton import JOINT_CHAIN, JOINT_NAMES

FORMAT_NAME = "msflow-motion"
FORMAT_VERSION = 1
NUM_JOINTS = 22
DEFAULT_FPS = 20  # HumanML3D fps, the webui's FRAMERATE


# webui/skeleton.py JOINT_CHAIN lacks the (12, 14) edge its left twin (12, 13)
# has; close that gap anatomically so the exported hierarchy is one tree
# (joint 14 currently only appears as a parent via (14, 17)).
_CHAIN_PARENT_GAPS: dict[int, int] = {14: 12}


def joint_parents() -> list[int]:
    """Parent joint per joint id (-1 for the pelvis), from JOINT_CHAIN."""
    parents = [-1] * NUM_JOINTS
    for parent, child in JOINT_CHAIN:
        parents[child] = parent
    for child, parent in _CHAIN_PARENT_GAPS.items():
        if parents[child] == -1:
            parents[child] = parent
    missing = [joint for joint in range(NUM_JOINTS) if joint and parents[joint] == -1]
    if missing:
        raise ValueError(f"joints without a parent in JOINT_CHAIN: {missing}")
    return parents


def build_motion_dict(
    positions: Any,
    *,
    fps: float = DEFAULT_FPS,
    controls: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize one motion as a schema-v1 `msflow-motion` dict.

    `positions` must be shaped (T, 22, 3) in MSFlow world space (Y-up, m).
    """
    world = np.asarray(positions, dtype=np.float32)
    if world.ndim != 3 or world.shape[1:] != (NUM_JOINTS, 3):
        raise ValueError(
            f"positions must be shaped (T, {NUM_JOINTS}, 3), got {world.shape}"
        )
    n_frames = int(world.shape[0])
    if n_frames <= 0:
        raise ValueError("positions must contain at least one frame.")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "fps": float(fps),
        "n_frames": n_frames,
        "duration_sec": round(n_frames / float(fps), 6),
        "joints": [JOINT_NAMES[i] for i in range(NUM_JOINTS)],
        "parents": joint_parents(),
        # 1e-6 m keeps files compact while staying far below float32 noise.
        "positions": [
            [[round(float(c), 6) for c in row] for row in frame] for frame in world
        ],
        "controls": list(controls or []),
        "meta": dict(meta or {}),
    }


def _load_run(run_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    samples_path = run_dir / "samples_world.npy"
    if not samples_path.is_file():
        raise FileNotFoundError(f"missing {samples_path}")
    samples_world = np.load(samples_path)
    if samples_world.ndim != 4 or samples_world.shape[2:] != (NUM_JOINTS, 3):
        raise ValueError(
            f"{samples_path} must be shaped (N, T, {NUM_JOINTS}, 3), "
            f"got {samples_world.shape}"
        )
    spec: dict[str, Any] = {}
    spec_path = run_dir / "spec.json"
    if spec_path.is_file():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return samples_world, spec


def export_sample(
    run_dir: str | Path,
    sample_index: int = 0,
    *,
    out_path: str | Path | None = None,
    fps: float = DEFAULT_FPS,
) -> Path:
    """Write `sample_<idx>.msflow.json` (or `out_path`) for one run sample."""
    run_dir = Path(run_dir)
    samples_world, spec = _load_run(run_dir)
    n_samples = int(samples_world.shape[0])
    if not 0 <= sample_index < n_samples:
        raise ValueError(f"sample_index must be in [0, {n_samples}), got {sample_index}.")
    meta = {
        "prompt": str(spec.get("prompt", "")),
        "source_motion_id": str(spec.get("source_motion_id", "")),
        "source_run": f"{run_dir.parent.name}/{run_dir.name}",
        "sample_index": int(sample_index),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    doc = build_motion_dict(
        samples_world[sample_index],
        fps=fps,
        controls=list(spec.get("controls") or []),
        meta=meta,
    )
    out = Path(out_path) if out_path else run_dir / f"sample_{sample_index:02d}.msflow.json"
    out.write_text(json.dumps(doc), encoding="utf-8")
    return out


def export_run(
    run_dir: str | Path,
    *,
    samples: list[int] | None = None,
    fps: float = DEFAULT_FPS,
) -> list[Path]:
    """Export every sample (or the given subset) of a run directory."""
    samples_world, _spec = _load_run(Path(run_dir))
    indices = list(range(int(samples_world.shape[0]))) if samples is None else samples
    return [
        export_sample(run_dir, index, fps=fps)  # type: ignore[arg-type]
        for index in indices
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a webui/CLI run directory to msflow-motion JSON."
    )
    parser.add_argument("--run", required=True, help="run dir with samples_world.npy")
    parser.add_argument(
        "--sample", type=int, default=-1, help="sample index; -1 exports all samples"
    )
    args = parser.parse_args(argv)
    run_dir = Path(args.run)
    paths = (
        export_run(run_dir)
        if args.sample < 0
        else [export_sample(run_dir, args.sample)]
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
