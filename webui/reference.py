"""Load HumanML3D motions as reference poses for the constraint editor."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from webui.skeleton import NUM_JOINTS


class ReferenceUnavailable(Exception):
    """Raised when no reference motion can be found for a given id."""


class ReferenceState:
    """Finds raw world-joint motions under the HumanML3D dataset copy and
    caches them per motion id.

    Search order per motion id:
      1. ``datasets/HumanML3D/joints/<id>.npy``       (raw world XYZ — preferred;
         handled for both [T, 22, 3] and [T, 3, 22] layouts)
      2. ``datasets/HumanML3D/motions/<id>.npy``      (raw world XYZ, same handling)
      3. ``datasets/HumanML3D/new_joints/<id>.npy``   (raw world XYZ [T, 22, 3],
         the layout HumanML3D ships)
      4. ``datasets/HumanML3D/new_joint_vecs/<id>.npy`` + recover_from_ric (263-d
         feature fallback; recovered coordinates are only a visual underlay and
         may not sit exactly in the raw model's world frame)
    """

    def __init__(self, data_root: Path | str) -> None:
        self.data_root = Path(data_root).resolve()
        self._cache: dict[str, dict[str, Any]] = {}

    @property
    def joints_dir(self) -> Path:
        return self.data_root / "joints"

    @property
    def motions_dir(self) -> Path:
        return self.data_root / "motions"

    @property
    def new_joints_dir(self) -> Path:
        return self.data_root / "new_joints"

    @property
    def feature_dir(self) -> Path:
        return self.data_root / "new_joint_vecs"

    @property
    def available(self) -> bool:
        return any(
            directory.is_dir()
            for directory in (
                self.joints_dir,
                self.motions_dir,
                self.new_joints_dir,
                self.feature_dir,
            )
        )

    def list_ids(self) -> list[str]:
        ids: set[str] = set()
        for directory in (
            self.joints_dir,
            self.motions_dir,
            self.new_joints_dir,
            self.feature_dir,
        ):
            if directory.is_dir():
                ids.update(p.stem for p in directory.glob("*.npy"))
        return sorted(ids)

    def get(self, motion_id: str) -> dict[str, Any]:
        """Return ``{"motion": [T, 22, 3] float32, "source": str}`` for a motion id."""
        motion_id = Path(motion_id).stem
        cached = self._cache.get(motion_id)
        if cached is not None:
            return cached

        raw = self._load_raw(motion_id)
        if raw is not None:
            entry = {"motion": raw, "source": "raw_world_joints"}
            self._cache[motion_id] = entry
            return entry

        recovered = self._load_features(motion_id)
        if recovered is not None:
            entry = {"motion": recovered, "source": "recovered_263d (visual only)"}
            self._cache[motion_id] = entry
            return entry

        raise ReferenceUnavailable(
            f"No reference motion '{motion_id}' under {self.data_root} "
            f"(checked joints/, motions/, new_joints/, new_joint_vecs/)."
        )

    def frame(self, motion_id: str, frame: int) -> tuple[np.ndarray, str]:
        """Reference pose [22, 3] at a frame index, clamped/modulo into range."""
        entry = self.get(motion_id)
        motion = entry["motion"]
        index = int(frame) % motion.shape[0] if motion.shape[0] else 0
        pose = np.asarray(motion[index], dtype=np.float32)
        return pose, str(entry["source"])

    # -- helpers ---------------------------------------------------------

    def _load_raw(self, motion_id: str) -> np.ndarray | None:
        for directory in (self.joints_dir, self.motions_dir, self.new_joints_dir):
            path = directory / f"{motion_id}.npy"
            if path.is_file():
                data = np.load(path, allow_pickle=False).astype(np.float32)
                motion = _as_tj3(data)
                if motion is not None:
                    return motion
        return None

    def _load_features(self, motion_id: str) -> np.ndarray | None:
        """Recover joints from 263-d features. recover_from_ric needs a torch
        tensor; any failure here degrades to None (canonical fallback)."""
        import torch

        path = self.feature_dir / f"{motion_id}.npy"
        if not path.is_file():
            return None
        data = np.load(path, allow_pickle=False).astype(np.float32)
        if data.ndim != 2 or data.shape[1] != 263:
            return None
        try:
            from utils.motion_process import recover_from_ric

            motion = np.asarray(
                recover_from_ric(torch.from_numpy(data), NUM_JOINTS),
                dtype=np.float32,
            )
        except Exception:
            return None
        return _as_tj3(motion)


def _as_tj3(data: np.ndarray) -> np.ndarray | None:
    """Normalize any raw layout to [T, 22, 3] world XYZ float32, else None."""
    array = np.asarray(data)
    if array.ndim == 3 and array.shape[1] == NUM_JOINTS and array.shape[2] == 3:
        return array.astype(np.float32)
    if array.ndim == 3 and array.shape[1] == 3 and array.shape[2] == NUM_JOINTS:
        return np.transpose(array, (0, 2, 1)).astype(np.float32)
    if array.ndim == 2:
        for columns, width in ((array.shape[1] // 3, NUM_JOINTS),):
            if columns == width:
                return array.reshape(array.shape[0], width, 3).astype(np.float32)
    return None
