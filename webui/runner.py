"""Model caching and generation plumbing for the webui.

Deliberately reuses the repo's own loaders/samplers so a GUI run is
byte-compatible with `sample/demo_joint_contorl.py` at the same seed:
the spec is written to a JSON snapshot and parsed by the *same*
`load_sparse_raw_control_spec` the CLI uses.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from os.path import join as pjoin
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

# isort: off
from utils.raw_inpaint_common import (
    dit_norm_to_world,
    load_sparse_raw_control_spec,
    raw_control_chw_to_dit,
    save_raw_inpaint_outputs,
)
from utils.raw_joint_utils import load_raw_joint_mean_std
from webui.spec_schema import safe_spec_name

# isort: on

ProgressCb = Callable[[float, str], None]

_SPEC_SNAPSHOT = "spec.json"


@dataclass
class RunSettings:
    name: str = "MMDiT_xyz_pretrained"
    ckpt: str = "latest"
    device: str = "cuda:0"
    dataset_name: str = "t2m"
    checkpoints_dir: str = "./checkpoints"
    dataset_dir: str = "./datasets"
    num_samples: int = 1
    num_steps: int = 300
    cfg: float = 2.0
    w_kin: float = 1.0
    seed: int = 3407
    use_metric_R: bool = True
    use_augmented_obs: bool = True
    use_noise_mixing: bool = True
    noise_mix_strength: float = 1.0
    save_mp4: bool = True
    save_gif: bool = False
    out_root: str = "generations/webui"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    out_dir: str
    videos: list[str] = field(default_factory=list)
    gifs: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    motion: Any = None  # [T, 22, 3] world joints of sample 0 (for preview)


class Runner:
    """Holds one model per checkpoint name/warm device; one run at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: torch.nn.Module | None = None
        self._model_key: tuple[str, str, str] | None = None

    def is_busy(self) -> bool:
        return self._lock.locked()

    # -- checkpoint discovery -------------------------------------------

    @staticmethod
    def list_checkpoints(
        checkpoints_dir: str = "./checkpoints", dataset_name: str = "t2m"
    ) -> list[dict[str, Any]]:
        base = Path(checkpoints_dir) / dataset_name
        found: list[dict[str, Any]] = []
        if not base.is_dir():
            return found
        for run_dir in sorted(base.iterdir()):
            if not (run_dir / "config.yaml").is_file():
                continue
            ckpts = (
                [p.stem for p in sorted((run_dir / "model").glob("*.tar"))]
                if (run_dir / "model").is_dir()
                else []
            )
            if not ckpts:
                continue
            found.append({"name": run_dir.name, "ckpts": ckpts})
        return found

    @staticmethod
    def list_devices() -> list[str]:
        devices = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(index)
                devices.append(f"cuda:{index} ({name})")
            devices[0] = "cuda:0"
        devices.append("cpu")
        return devices

    # -- generation ------------------------------------------------------

    def run(
        self, spec: dict, settings: RunSettings, progress_cb: ProgressCb | None = None
    ) -> RunResult:
        """Run one generation from a spec *json dict* (loader schema)."""
        os.environ.setdefault("MPLBACKEND", "Agg")

        with self._lock:
            return self._run_locked(spec, settings, progress_cb)

    def _run_locked(
        self, spec: dict, settings: RunSettings, progress_cb: ProgressCb | None
    ) -> RunResult:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = safe_spec_name(spec.get("spec_name") or "run")
        out_dir = Path(settings.out_root) / run_name / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        spec_path = out_dir / _SPEC_SNAPSHOT
        saved_spec = {key: value for key, value in spec.items() if key != "spec_name"}
        spec_path.write_text(json.dumps(saved_spec, indent=2), encoding="utf-8")

        device = self._resolve_device(settings.device)

        mean, std = load_raw_joint_mean_std(
            settings.dataset_name,
            data_root=pjoin(settings.dataset_dir, "HumanML3D"),
        )
        (
            control_chw,
            mask_chw,
            control_world,
            joint_ids,
            prompt,
            metadata,
        ) = load_sparse_raw_control_spec(
            str(spec_path),
            mean,
            std,
            n_joints=22,
        )
        argv = _argparse_stub(settings)
        argv.text = prompt

        model = self._load_model(settings, device, progress_cb)
        if progress_cb:
            progress_cb(0.05, f"model ready on {device}; sampling …")

        total_calls: list[int] = [0]
        original_forward = model.forward_with_CFG

        def counted_forward(*call_args: Any, **call_kwargs: Any) -> torch.Tensor:
            out = original_forward(*call_args, **call_kwargs)
            total_calls[0] += 1
            if progress_cb:
                fraction = min(total_calls[0] / max(settings.num_steps - 1, 1), 1.0)
                progress_cb(
                    0.05 + 0.9 * fraction, f"step {total_calls[0]}/{settings.num_steps}"
                )
            return out

        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)

        model.forward_with_CFG = counted_forward  # type: ignore[method-assign]
        try:
            from sample.demo_joint_contorl import sample_with_raw_dit_inpaint

            control_dit, mask_dit = raw_control_chw_to_dit(control_chw, mask_chw)
            samples_norm = sample_with_raw_dit_inpaint(
                model,
                prompt=prompt,
                control=control_dit,
                mask=mask_dit,
                length=int(control_world.shape[0]),
                num_samples=settings.num_samples,
                cfg=settings.cfg,
                num_steps=settings.num_steps,
                seed=settings.seed,
                use_metric_R=settings.use_metric_R,
                use_augmented_obs=settings.use_augmented_obs,
                use_noise_mixing=settings.use_noise_mixing,
                noise_mix_strength=(
                    settings.noise_mix_strength
                    if settings.noise_mix_strength is not None
                    and settings.use_noise_mixing
                    else None
                ),
                w_kin=settings.w_kin,
                **dict(settings.extras),
            )
        finally:
            model.__dict__.pop("forward_with_CFG", None)

        samples_norm_tj3, samples_world = dit_norm_to_world(
            samples_norm,
            mean,
            std,
            n_joints=22,
        )
        argv.out_dir = str(out_dir)
        argv.save_mp4 = settings.save_mp4
        argv.save_gif = settings.save_gif
        save_raw_inpaint_outputs(
            argv,
            samples_norm_tj3=samples_norm_tj3,
            samples_world=samples_world,
            control_world=control_world,
            control_mask_chw=mask_chw,
            trajectory_requested=control_world[:, list(joint_ids)],
            trajectory_used=control_world[:, list(joint_ids)],
            joint_ids=joint_ids,
            test_motion_id=None,
            test_prompt_index=None,
            sparse_control_metadata=metadata,
        )
        videos = (
            [
                str(out_dir / f"sample_{i:02d}.mp4")
                for i in range(samples_world.shape[0])
            ]
            if settings.save_mp4
            else []
        )
        gifs = (
            [
                str(out_dir / f"sample_{i:02d}.gif")
                for i in range(samples_world.shape[0])
            ]
            if settings.save_gif
            else []
        )
        files = [str(spec_path)] + [
            str(out_dir / nm) for nm in sorted(_list_out_files(out_dir))
        ]
        if progress_cb:
            progress_cb(1.0, "done")
        motion = None
        try:
            motion = samples_world[0].tolist()
        except Exception:
            motion = None
        return RunResult(
            out_dir=str(out_dir),
            videos=videos,
            gifs=gifs,
            files=files,
            motion=motion,
        )

    # -- internals -------------------------------------------------------

    def _resolve_device(self, device: str) -> torch.device:
        head = device.split(" ")[0]
        if head.startswith("cuda") and torch.cuda.is_available():
            return torch.device(head)
        return torch.device("cpu")

    def _model_args(self, settings: RunSettings) -> argparse.Namespace:
        return argparse.Namespace(
            dataset_name=settings.dataset_name,
            dataset_dir=settings.dataset_dir,
            checkpoints_dir=settings.checkpoints_dir,
            name=settings.name,
            ckpt=settings.ckpt,
            checkpoint_key="ema_model",
            gpu="0",
            n_frames=1024,
        )

    def _load_model(
        self,
        settings: RunSettings,
        device: torch.device,
        progress_cb: ProgressCb | None,
    ) -> torch.nn.Module:
        from utils.raw_inpaint_model import load_model

        key = (settings.name, settings.ckpt, settings.device)
        if self._model_key == key and self._model is not None:
            model = self._model.to(device)
        else:
            if progress_cb:
                progress_cb(0.02, f"loading {settings.name}/{settings.ckpt} …")
            model, _ = load_model(self._model_args(settings), device=device)
            self._model = model
            self._model_key = key
        return model


def _list_out_files(out_dir: Path) -> list[str]:
    return sorted(
        p.name for p in out_dir.iterdir() if p.is_file() and p.name != _SPEC_SNAPSHOT
    )


def _argparse_stub(settings: RunSettings) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_name=settings.dataset_name,
        dataset_dir=settings.dataset_dir,
        checkpoints_dir=settings.checkpoints_dir,
        name=settings.name,
        ckpt=settings.ckpt,
        checkpoint_key="ema_model",
        gpu=settings.device,
        n_frames=0,
        num_samples=settings.num_samples,
        cfg=settings.cfg,
        num_steps=settings.num_steps,
        seed=settings.seed,
        out_dir="",
        save_mp4=False,
        save_gif=False,
    )
