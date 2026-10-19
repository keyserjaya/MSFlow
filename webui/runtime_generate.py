"""Text-to-motion sampling core for the Unity runtime server.

Factored from ``sample/demo_msflow_263.py`` so the runtime bridge uses the
exact same checkpoint loader (``utils.raw_inpaint_model.load_model``) as the
webui runner — one loaded model serves both the plain-text sampler
(``model.generate``) and the sparse constraint sampler (``Runner.run``).
Pure torch/numpy: no matplotlib, no mp4, no hydra.
"""
from __future__ import annotations

import argparse
import random
import threading
from dataclasses import dataclass
from os.path import join as pjoin
from typing import Any, Callable

import numpy as np
import torch

from utils.raw_inpaint_common import dit_norm_to_world
from utils.raw_inpaint_model import load_model
from utils.raw_joint_utils import load_raw_joint_mean_std

ProgressCb = Callable[[float, str], None]


@dataclass
class ModelSettings:
    """Checkpoint/paths for one model instance (mirrors Runner settings)."""

    name: str = "MMDiT_xyz_pretrained"
    ckpt: str = "latest"
    checkpoint_key: str = "ema_model"
    device: str = "cuda:0"
    dataset_name: str = "t2m"
    dataset_dir: str = "./datasets"
    checkpoints_dir: str = "./checkpoints"

    def model_args(self) -> argparse.Namespace:
        # n_frames=1024 matches Runner._model_args: the loading-time guard is
        # bypassed here; the server validates the requested length itself.
        return argparse.Namespace(
            dataset_name=self.dataset_name,
            dataset_dir=self.dataset_dir,
            checkpoints_dir=self.checkpoints_dir,
            name=self.name,
            ckpt=self.ckpt,
            checkpoint_key=self.checkpoint_key,
            gpu="0",
            n_frames=1024,
        )


class ModelCache:
    """One checkpoint per (name, ckpt, device); thread-safe.

    The Unity runtime server keeps a single cache instance shared between the
    text-only sampler and the constraint Runner so both modes reuse the same
    VRAM. ``max_length`` reports the model's sequence limit.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: torch.nn.Module | None = None
        self._key: tuple[str, str, str] | None = None

    def get(self, settings: ModelSettings, device: torch.device) -> torch.nn.Module:
        key = (settings.name, settings.ckpt, str(device))
        with self._lock:
            if self._model is not None and self._key == key:
                return self._model.to(device)
            model, _ = load_model(settings.model_args(), device=device)
            self._model = model
            self._key = key
            return model

    def max_length(self, settings: ModelSettings, device: torch.device) -> int:
        model = self.get(settings, device)
        return int(getattr(model, "max_length", 1024))


@dataclass(frozen=True)
class TextGenerationResult:
    positions: np.ndarray  # [T, 22, 3] world joints (Y-up, metres), 20 fps
    n_frames: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TextMotionGenerator:
    """Plain prompt -> [T, 22, 3] world joints (the demo_msflow path, batch 1)"""

    def __init__(self, cache: ModelCache, settings: ModelSettings) -> None:
        self._cache = cache
        self._settings = settings

    def resolve_device(self) -> torch.device:
        head = self._settings.device.split(" ")[0]
        if head.startswith("cuda") and torch.cuda.is_available():
            return torch.device(head)
        return torch.device("cpu")

    def max_length(self) -> int:
        return self._cache.max_length(self._settings, self.resolve_device())

    def generate(
        self,
        prompt: str,
        *,
        n_frames: int,
        seed: int = 3407,
        cfg: float = 3.0,
        progress_cb: ProgressCb | None = None,
    ) -> TextGenerationResult:
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt must be a non-empty string.")
        if n_frames <= 0:
            raise ValueError(f"n_frames must be positive, got {n_frames}.")

        settings = self._settings
        device = self.resolve_device()
        if progress_cb:
            progress_cb(0.02, f"loading {settings.name}/{settings.ckpt} …")
        model = self._cache.get(settings, device)

        max_length = int(getattr(model, "max_length", 1024))
        if n_frames > max_length:
            raise ValueError(
                f"n_frames must be <= model max_length ({max_length}), "
                f"got {n_frames}."
            )

        seed_everything(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        data_root = pjoin(settings.dataset_dir, "HumanML3D")
        mean, std = load_raw_joint_mean_std(settings.dataset_name, data_root=data_root)

        total_calls = [0]
        sampling_timesteps = int(getattr(model, "sampling_timesteps", 50))
        original_forward = model.forward_with_CFG

        def counted_forward(*call_args: Any, **call_kwargs: Any) -> torch.Tensor:
            out = original_forward(*call_args, **call_kwargs)
            total_calls[0] += 1
            if progress_cb:
                fraction = min(total_calls[0] / max(sampling_timesteps - 1, 1), 1.0)
                progress_cb(
                    0.05 + 0.9 * fraction,
                    f"step {total_calls[0]}/{sampling_timesteps}",
                )
            return out

        model.forward_with_CFG = counted_forward  # type: ignore[method-assign]
        try:
            with torch.no_grad():
                length = torch.tensor([n_frames], dtype=torch.long, device=device)
                pred_motions = model.generate(
                    [str(prompt)], length, cond_scale=cfg
                )
        finally:
            model.__dict__.pop("forward_with_CFG", None)

        # generate() returns [B, T, 1, 66] (batch 1 here); dit_norm_to_world
        # slices the padded axis, unflattens to (B, T, 22, 3) and denormalizes.
        if pred_motions.ndim != 4 or tuple(pred_motions.shape[2:]) != (1, 66):
            raise ValueError(
                "Text-only generation expects the XYZ checkpoint (66-dim raw "
                f"joints), got motion shape {tuple(pred_motions.shape)}."
            )
        _, samples_world = dit_norm_to_world(pred_motions, mean, std, n_joints=22)
        world = np.asarray(samples_world[0], dtype=np.float32)
        if world.shape[1:] != (22, 3):
            raise ValueError(
                f"denormalized motion must be shaped (T, 22, 3), got {world.shape}"
            )
        if progress_cb:
            progress_cb(1.0, "done")
        return TextGenerationResult(positions=world, n_frames=int(world.shape[0]))
