"""Parity check: webui runner vs sample/demo_joint_contorl.py at the same seed.

    uv run --group ui python webui/parity_check.py --spec kick --gpu cpu --num_steps 100

Exit code 0 = parity (or tiny float drift), non-zero = divergence.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BUNDLED = ROOT / "sample" / "joint_control_specs"


def run_cli(args: argparse.Namespace, out_dir: Path, spec_path: Path) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "sample.demo_joint_contorl",  # module form matches the repo's scripts
        "--name",
        args.name,
        "--ckpt",
        args.ckpt,
        "--gpu",
        args.gpu,
        "--sparse_control_spec",
        str(spec_path),
        "--out_dir",
        str(out_dir),
        "--num_samples",
        "1",
        "--num_steps",
        str(args.num_steps),
        "--cfg",
        str(args.cfg),
        "--w_kin",
        str(args.w_kin),
        "--noise_mix_strength",
        "1",
        "--seed",
        str(args.seed),
    ]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
        timeout=1800,
    )
    if completed.returncode != 0:
        print(completed.stdout[-2000:])
        print(completed.stderr[-3000:])
        raise SystemExit("CLI run failed")
    expected = out_dir / "motion_raw_norm.npy"
    if not expected.is_file():
        print(completed.stdout[-2000:])
        raise SystemExit("CLI produced no motion_raw_norm.npy")
    return expected


def run_webui(args: argparse.Namespace, out_dir: Path, spec_path: Path) -> Path:
    from webui.runner import Runner, RunSettings
    from webui.spec_schema import load_spec_file

    spec = load_spec_file(spec_path)
    payload = spec.to_json_dict()

    def cb(fraction: float, message: str) -> None:
        print(f"  [{int(fraction * 100):3d}%] {message}")

    runner = Runner()
    result = runner.run(
        payload,
        RunSettings(
            name=args.name,
            ckpt=args.ckpt,
            device=args.gpu,
            num_samples=1,
            num_steps=args.num_steps,
            cfg=args.cfg,
            w_kin=args.w_kin,
            seed=args.seed,
            save_mp4=False,
            save_gif=False,
            out_root=str(out_dir),
        ),
        cb,
    )
    produced = Path(result.out_dir) / "motion_raw_norm.npy"
    if not produced.is_file():
        raise SystemExit(f"webui produced no motion_raw_norm.npy in {result.out_dir}")
    return produced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="kick")
    parser.add_argument("--name", default="MMDiT_xyz_pretrained")
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--gpu", default="cpu")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--w_kin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.is_file():
        spec_path = BUNDLED / f"{args.spec}.json"
    if not spec_path.is_file():
        raise SystemExit(f"spec not found: {args.spec}")

    with tempfile.TemporaryDirectory(prefix="parity_") as tmp:
        tmp_path = Path(tmp)
        cli_npy = run_cli(args, tmp_path / "cli", spec_path)
        webui_npy = run_webui(args, tmp_path / "webui", spec_path)
        a = np.load(cli_npy)
        b = np.load(webui_npy)
        if a.shape != b.shape:
            raise SystemExit(f"shape mismatch: cli {a.shape} vs webui {b.shape}")
        if np.array_equal(a, b):
            print("parity OK: bit-identical on cpu")
            return
        deviation = float(np.max(np.abs(a - b)))
        print(f"parity drift: max |Δ| = {deviation:.3g}")
        raise SystemExit(0 if deviation < 2e-3 else 1)


if __name__ == "__main__":
    main()
