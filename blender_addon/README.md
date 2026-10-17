# MSFlow Motion Bridge (Blender add-on)

Receives motion exported from the MSFlow webUI/CLI and retargets it onto any
armature as a new Action.

## Install

1. Zip the `msflow_bridge` folder (the folder itself, so the zip contains
   `msflow_bridge/__init__.py`), or point Blender's add-on path at
   `blender_addon/` and enable **Add-ons ▸ Animation ▸ MSFlow Motion Bridge**
   (Blender 4.2 LTS or newer).
2. Open the 3D-View sidebar (N) ▸ **MSFlow** tab.

## Use with the webUI

1. In the webUI, generate a run. Every sample is exported next to its npy/mp4
   outputs as `sample_XX.msflow.json` (schema: `msflow-motion` v1, 20 fps,
   world joints + the sparse-control spec).
2. In Blender, pick your armature in the panel, then either
   **Import MSFlow motion** (file browser), **Latest run**, turn on the
   **folder watch** (auto-import as runs land — WSL2-safe, no port), or click
   **Send to Blender** in the webUI (needs the push receiver running here).
3. Bone mapping: **Auto-match bones** fills the 22 slots from bone names
   (Rigify/Mixamo/UE-style names both work; MCH/WGT decoys are ignored).
   **Match by closest head** derives it from rest-pose geometry instead.
   Save/load mappings as JSON presets.
4. Options: root motion on the hips bone / armature object / in place,
   leg-scale matching, flip facing (if the character walks the wrong way),
   integer-frame keyframe rounding, scene frame-range fitting, muted **MSFlow
   NLA clip**, constraint-target empties.

## Notes

- The source is 22 joint *positions* per frame; bone roll is reconstructed
  from chain geometry (possible twist drift on highly rolled poses).
- Keys are placed at `1 + i * scene_fps / 20` so the clip plays in real
  seconds at any scene frame rate; the scene fps is never modified.
- The mapped minimum is: pelvis + one spine slot + both leg chains + both
  upper arms; unmapped joints keep their rest pose (fingers stay put).
- Headless check (needs a `blender` executable on PATH):
  `blender --background --python blender_addon/tests/retarget_selftest.py`
