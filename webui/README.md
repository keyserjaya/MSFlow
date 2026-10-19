# MSFlow Constraint WebUI

A browser editor for the sparse frame/joint constraints used by
`sample/demo_joint_contorl.py` (the paper's zero-shot spatial control:
dragging/walking/kicking driven by any-frame, any-joint XYZ targets), plus a
launcher for the projection sampler.

## Run

From the repo root inside WSL2 (the repo's own environment):

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/venvs/msflow"
uv run --group ui python webui/app.py
```

Open the printed URL (default <http://localhost:7860>) from the Windows-side
browser — WSL2 forwards localhost automatically. If not, use
`0.0.0.0`-reachable LAN URL printed by Gradio.

The UI dependency group stays out of core research workflows: it is only
installed for `--group ui` invocations.

## Editor concepts

- **Current frame** — one slider drives everything: the armature preview, the
  reference underlay (when a motion id is set, its pose is sampled at this
  frame) and where edits land. Selecting a keyframe jumps the slider to its
  frame.
- **3D scene editor (left)** — a Unity-style viewport (three.js): orbit with
  left-drag, pan with right-drag, zoom with scroll. Every saved constraint is
  a colored handle: solid at the current frame, a faint ghost at other frames
  (clicking a ghost jumps there). Click any joint or handle to select it
  client-side (no refresh flash); a translate gizmo (Unity's move tool)
  appears — drag an arrow to move along one axis, or drag a plane handle to
  slide on a plane. Moved joints are added to a keyframe at the current frame
  and the touched axes become constrained automatically (still tunable via
  the axes checkboxes). Esc deselects.
- **IK armature preview** — a lightweight CCD solver in the browser poses the
  skeleton toward the current frame's constraints every time you drag, so the
  preview armature matches the placed constraints ("estimated result"; the
  generated motion still comes from the model, which honors XYZ positions
  only).
- **🧭 Waypoints** — pelvis path pins (joint 0, x/z only): the root passes
  through point A then point B while height, torso and legs stay animated
  (use these instead of spine constraints — pinning a spine joint drags the
  torso rigidly and looks stiff). To move one: click the waypoint ball (it is
  drawn larger and through the body) — the gizmo arms immediately; drag the
  X/Z arrows, or drag the **square** (XZ plane handle) to slide it along the
  ground. Clicking a selected marker never collapses the gizmo; deselect with
  Esc or a click on empty space. Whether the marker is a ghost at another
  frame, one click jumps the editing frame there with the gizmo armed.
  Repeated 🧭 clicks select the placed waypoint; two or more draw the path
  line through the ordered pins.
- **🎯 Constrain this frame** — button in the viewport: captures the whole
  solved pose at the current frame (all 22 joints, x/y/z) as constraints,
  replacing that frame's keyframe.
- **Result preview** — after a generation, the sampled motion plays back in
  the same viewport as an orange skeleton, with a play/scrub bar (top-right of
  the viewport). All saved constraints stay visible as transparent ghosts so
  you can see how the result relates to the placed targets.
- **Duration** — set the motion length in seconds (or directly as `n_frames`;
  the two stay in sync at 20 fps).
- **Reference motion (right)** — pick a HumanML3D motion id; the canvas draws
  the referenced pose (at the *current frame*) as an underlay, and new
  keyframes / newly controlled joints start from those positions. Falls back
  to a canonical standing pose when the dataset/id is missing.
- **Keyframes** — each row constrains a set of joints at one frame with a
  chosen `axes` subset (`x,y,z` or fewer). Stage colors match the CLI's
  early/middle/late palette.
- **Run** — same flags as the CLI (`num_steps`, `cfg`, `w_kin`, `seed`,
  `num_samples`, `use_metric_R`, `use_augmented_obs`, `use_noise_mixing`,
  `noise_mix_strength`); outputs (npy/xz png/mp4) land in
  `generations/webui/<spec>/<timestamp>/`, videos render inline, everything
  is bundled as a zip.

Saved specs are JSON files under `specs/`, round-trip-safe with the bundled
`sample/joint_control_specs/*.json` and the CLI loader — anything authored
here can be replayed via the CLI command:

```bash
uv run python sample/demo_joint_contorl.py --name MMDiT_xyz_pretrained \
    --ckpt latest --sparse_control_spec specs/mine.json --out_dir tmp/out \
    --num_steps 300 --cfg 2 --w_kin 1 --save_mp4
```

## Components

| File | Purpose |
| --- | --- |
| `app.py` | Gradio Blocks UI + event wiring |
| `runner.py` | model cache, spec→sample plumbing, progress, output saving |
| `spec_schema.py` | spec parse/serialize (round-trips the CLI loader) |
| `reference.py` | HumanML3D reference-pose loader (raw joints → feature fallback) |
| `skeleton.py` | 22-joint names/chain + canonical fallback pose |
| `components/constraint_canvas/` | custom Gradio component (three.js 3D scene editor + move gizmo) |

## Building the drag canvas component

One-time (Node required):

```bash
sudo apt install -y nodejs npm
uv run --group ui gradio cc build webui/components/constraint_canvas
```

Until the component is built, the app still runs: the canvas area shows a
placeholder and all editors are numeric-only.

## Parity with the CLI

`webui/parity_check.py` runs the same spec through the webui runner and the
CLI demo script and compares the resulting `motion_raw_norm.npy` arrays:

```bash
uv run --group ui python webui/parity_check.py --spec kick --gpu cpu --num_steps 100
```

(CUDA/tf32 introduces tiny float drift; use `--gpu cpu` for exact equality.)

## Blender bridge

Every run also writes `sample_XX.msflow.json` next to its other outputs
(schema `msflow-motion` v1: 20 fps, `[T, 22, 3]` world joints, the sparse
controls, and meta). Use `python -m webui.msflow_export --run
generations/webui/<spec>/<timestamp>` to (re)export older runs. In Blender,
install the small add-on under `blender_addon/` (see its README) to import a
run, retarget it onto your armature, and land it as a new Action (optionally
a muted NLA clip), with the constraint targets drawn as empties. The Run
panel's **send to Blender** button pushes the files to the add-on's receiver
over `127.0.0.1` (under WSL2 NAT also enable "bind all" + "bind all
interfaces" in Blender, or just keep the add-on's folder watch on — it polls
the shared output root). Export-side tests: `uv run --group ui python
webui/test_msflow_export.py`.
