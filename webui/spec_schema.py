"""Parse/serialize MSFlow sparse-control specs (round-trip safe with the CLI loader)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from webui.skeleton import JOINT_NAMES

# Reuse the exact palette the CLI uses for stage labels.
STAGE_COLORS: dict[str, str] = {
    "early": "#0072B2",
    "middle": "#CC79A7",
    "late": "#009E73",
}
AXES_MAP: dict[str, int] = {"x": 0, "y": 1, "z": 2}
AXES_BY_INDEX: tuple[str, ...] = ("x", "y", "z")


@dataclass
class Control:
    """One keyframe constraint. `values` is always per-joint world XYZ
    (shape (len(joints), 3)), matching the CLI loader. Only the selected
    axes are masked as constraints; the rest ride along harmlessly."""

    frame: int
    joints: list[int]
    axes: list[int]
    values: list[list[float]]
    stage: str = ""
    frame_fraction: float | None = None
    # Original axes spelling, preserved so save→load round-trips byte-identically.
    axes_raw: str | None = None

    def __post_init__(self) -> None:
        n = len(self.joints)
        if len(self.values) != n or any(len(row) != 3 for row in self.values):
            raise ValueError(
                f"Control at frame {self.frame} needs values shaped ({n}, 3)."
            )

    def to_dict(self) -> dict:
        out: dict = {"joints": [int(j) for j in self.joints]}
        if self.frame_fraction is not None:
            out["frame_fraction"] = float(self.frame_fraction)
        else:
            out["frame"] = int(self.frame)
        out["axes"] = (
            self.axes_raw
            if self.axes_raw
            else ",".join(AXES_BY_INDEX[a] for a in self.axes)
        )
        out["values"] = [[float(v) for v in row] for row in self.values]
        if self.stage:
            out["stage"] = self.stage
        return out


@dataclass
class Spec:
    """A full sparse-control spec. `controls` order is meaningful for display
    only; the loader treats entries independently."""

    n_frames: int = 120
    prompt: str = ""
    source_motion_id: str = ""
    controls: list[Control] = field(default_factory=list)
    joint_names: dict[str, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)  # any extra unknown keys preserved

    def to_json_dict(self) -> dict:
        out: dict = {
            "prompt": self.prompt,
            "n_frames": int(self.n_frames),
        }
        if self.source_motion_id:
            out["source_motion_id"] = self.source_motion_id
        names = self.joint_names or {str(k): v for k, v in JOINT_NAMES.items()}
        out["joint_names"] = {str(k): v for k, v in names.items()}
        out["controls"] = [control.to_dict() for control in self.controls]
        if self.meta:
            for key, value in self.meta.items():
                out.setdefault(key, value)
        return out

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_json_dict(), indent=2), encoding="utf-8"
        )


def parse_spec(raw: dict) -> Spec:
    """Mirror utils.raw_inpaint_common.load_sparse_raw_control_spec validation."""
    n_frames = int(raw["n_frames"])
    if n_frames <= 0:
        raise ValueError("Sparse control spec n_frames must be positive.")
    prompt = str(raw["prompt"])
    spec = Spec(n_frames=n_frames, prompt=prompt)
    spec.source_motion_id = str(raw.get("source_motion_id", ""))
    spec.joint_names = {
        str(k): str(v) for k, v in (raw.get("joint_names") or {}).items()
    }
    controls = raw.get("controls", [])
    if not controls:
        raise ValueError("Sparse control spec must contain at least one control.")
    spec.meta = {
        key: value
        for key, value in raw.items()
        if key
        not in {"prompt", "n_frames", "source_motion_id", "joint_names", "controls"}
    }
    for entry in controls:
        has_frame = "frame" in entry
        has_fraction = "frame_fraction" in entry
        if has_frame == has_fraction:
            # Exactly one of frame / frame_fraction (same rule as the loader).
            raise ValueError(
                "Each sparse control needs exactly one of frame or frame_fraction."
            )
        if has_frame:
            frame = int(entry["frame"])
            fraction = None
        else:
            fraction = float(entry["frame_fraction"])
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("frame_fraction must be in [0, 1].")
            frame = int(round(fraction * (n_frames - 1)))
        if not 0 <= frame < n_frames:
            raise ValueError(f"Control frame must be in [0, {n_frames}), got {frame}.")
        if not entry["joints"]:
            raise ValueError("At least one joint id is required.")
        joints, seen = [], set()
        for j in [int(item) for item in entry["joints"]]:
            if not 0 <= j < 22:
                raise ValueError(f"Joint ids must be in [0, 22), got {j}.")
            if j not in seen:
                seen.add(j)
                joints.append(j)
        axes = parse_axes(str(entry.get("axes", "x,y,z")))
        axes_raw = (
            str(entry["axes"])
            if "axes" in entry
            else ",".join(AXES_BY_INDEX[a] for a in axes)
        )
        values = [[float(v) for v in row] for row in entry["values"]]
        control = Control(
            frame=frame,
            joints=joints,
            axes=axes,
            values=values,
            stage=str(entry.get("stage", "")),
            frame_fraction=fraction,
            axes_raw=axes_raw,
        )
        spec.controls.append(control)
    return spec


def parse_axes(spec: str) -> list[int]:
    spec = (spec or "x,y,z").replace(" ", "")
    try:
        axes = [AXES_MAP[item] for item in spec.split(",") if item]
    except KeyError as exc:
        raise ValueError(f"Unknown axis in '{spec}'. Use x,y,z.") from exc
    return list(dict.fromkeys(axes))


def empty_spec(n_frames: int = 120, prompt: str = "") -> Spec:
    spec = Spec(n_frames=n_frames, prompt=prompt)
    spec.joint_names = {str(k): v for k, v in JOINT_NAMES.items()}
    return spec


def load_spec_file(path: str | Path) -> Spec:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return parse_spec(raw)


def safe_spec_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", name).strip("_") or "untitled"
