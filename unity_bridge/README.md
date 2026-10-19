# MSFlow Motion Bridge (Unity runtime)

Generate HumanML3D motion from text prompts — and sparse joint controls — on a
running MSFlow server, and drive a **Humanoid** Animator character with the
result at Unity runtime. This package is the Unity counterpart of
`blender_addon/`: the same `msflow-motion` v1 documents flow to either
consumer.

```
Unity (play mode)                     WSL2 (uv venv, CUDA)
MsflowSocketClient ── WebSocket ────►  webui/runtime_server.py
  "person waving hand"                  ├─ text-only → model.generate(...)
  ◄── progress / result ──────────────  └─ controls → webui.runner.Runner
MsflowMotion parse → MsflowRetarget
  (positions → bone rotations)      generations/runtime/.../*.msflow.json
MsflowPlaybackDriver.LateUpdate      (the Blender folder-watch sees these too)
```

## 1. Requirements

- Unity **2021.3 LTS or newer** (developed against Unity 6), Built-in or URP.
- The MSFlow repo with the WSL2 `uv` environment (`deploy.md`) and the
  `checkpoints/t2m/MMDiT_xyz_pretrained` checkpoint.
- `websockets` in the server env (installed automatically by the command below).

## 2. Install the package

Add a local package entry to your project's `Packages/manifest.json`:

```json
"com.msflow.motion-bridge": "file:G:\\AIProjects\\MSFlow\\unity_bridge"
```

or copy the `unity_bridge/` folder into your project's `Packages/` directory.

## 3. Start the server (WSL2)

```bash
cd /mnt/g/AIProjects/MSFlow
uv run --group server python webui/runtime_server.py --port 8766 --no-save-mp4
```

Useful flags: `--bind 0.0.0.0` (WSL2 NAT setups; default is loopback, which
Windows reaches via WSL2 localhost forwarding), `--device cpu` (no GPU),
`--name MMDiT_xyz_pretrained --ckpt latest` (default), `--out-root
generations/runtime` (default; artifacts land here and the Blender add-on's
"Import latest run" picks them up).

Protocol (v1, JSON over one WebSocket at `ws://127.0.0.1:8766`):

| client → server | server → client |
| --- | --- |
| `hello {protocol:1}` | `hello {protocol, fps, max_length, model, ckpt}` |
| `generate {job_id?, prompt, duration_sec\|n_frames, seed?, cfg?, num_steps?, controls?}` | `accepted {job_id, queue_position}` |
| `fetch {job_id}` (after reconnect) | `progress {job_id, fraction, message}` |
| `cancel {job_id}` | `result {job_id, motion:<msflow-motion v1>}` / `error {job_id, message}` |

Omitting `controls` gives plain text-to-motion; sending the webUI spec-loader
control entries (`{frame|frame_fraction, joints, axes, values}`) runs the
constraint sampler. Results are cached server-side (last 8 jobs / 5 min), so a
dropped connection does not lose a finished generation — reconnect and `fetch`.

## 4. Use in your scene

Either use the sample (menu **MSFlow → Create Basic Sample Scene**; import the
package's Samples first) or wire it manually:

1. Add `MsflowPlaybackDriver` to (or under) the object with your humanoid
   `Animator`.
2. Add `MsflowMotionRequest` anywhere and point `ServerUrl` at the bridge.
3. Generate:

```csharp
var motion = await request.GenerateAsync("person waving hand");
// AutoPlay=true already played it on the driver; otherwise:
driver.Play(motion);
```

The sample demo defaults to **3-second clips** (shorter prompts produce tamer,
gesture-like motion) and a **`PinStance` preset** — pelvis XZ + both ankles
pinned at the first and last frame via sparse controls, so the gesture stays
grounded instead of turning into a full-body performance. Toggle it off for
free generation, or send your own controls:

```csharp
var controls = new List<MsflowControl>
{
    new MsflowControl { Frame = 30, Joints = new[] { 20 },
        Values = new[] { new[] { 0.5f, 1.2f, 0.3f } } },
};
await request.GenerateAsync("a person kicks forward", 4f, controls, numSteps: 100);
```

`numSteps` is the constraint sampler's flow-step count (server default 300;
100 is a good interactive compromise). Plain prompt requests use the fast
text path (50 fixed steps).

## 5. Constraint authoring (`MsflowConstraintBuilder`)

Building wire controls by hand means knowing the 22-joint indices
(`MsflowSkeleton`), the data convention (X-mirrored relative to Unity), and
the canonical figure scale. `MsflowConstraintBuilder` takes Unity intent
instead and converts per pin:

```csharp
var b = MsflowConstraintBuilder.ForCharacter(animator);   // or ForDriver(driver)
b.PinBone(HumanBodyBones.LeftFoot, frameFraction: 0.5f, worldPosition);
b.PinBoneThroughout(HumanBodyBones.Hips, intervalFrames: 10); // hold current pos
b.PinCurrentPoseAtStart();                  // snapshot the live pose at frame 0
var controls = b.Build();
await request.GenerateAsync("person waving hand", 3f, controls);
// or let the request expand the builder against the clip:
await request.GenerateAsync("person waving hand", 3f, b, numSteps: 100);
```

- **Positions only.** The wire protocol pins joint *positions* (optionally per
  axis, `axes: "x,z"`); rotation constraints are not expressible.
- **Capture timing.** Values are captured when the pin method is called (the
  live pose for `PinPose` / `PinBoneThroughout` / `Pin*AtStart*`), so results
  are deterministic. Pins are keyed by (joint, frame fraction ±1e-3) and a
  later pin replaces an earlier one; `Build()` emits one control per
  (fraction, axes) group. `PinPose` skips joints without a bone (one
  aggregated warning); unknown bones throw listing the supported set.
- **Scale normalization.** A pin at a world position converts as
  `data = FromDataConvention(root.worldToLocalMatrix * pos) / scale`, where
  `scale` is the character's hip→ankle span over the canonical 0.88 m
  (`NormalizeScale = false` sends raw coordinates; a degenerate span falls
  back to 1). Root-local conversion keeps pins stable when the character
  moves between authoring and generation, and the X-flip involution makes
  them round-trip exactly through the engine's output mapping.
- **Clip positions** are `frameFraction` (0 = first frame, 1 = last). The
  integer-frame form stays available on `MsflowControl` for hand-built lists.
  `PinBoneThroughout` expands to every `intervalFrames`-th frame of a
  `FrameCount`-frame clip (the `GenerateAsync` builder overloads set
  `FrameCount` from the requested duration at 20 fps; set it yourself before
  `Build()` otherwise).
- **`PinStartPose`** on `MsflowMotionRequest` (default off) snapshots the
  target driver's live pose at frame 0 and merges it *under* the caller's
  controls — caller pins win on duplicates — so a generated clip starts from
  the pose the character is actually in.

## 6. How the motion is applied

- The 22 source joints map onto `HumanBodyBones` (pelvis→Hips, hips→UpperLeg,
  knees→LowerLeg, ankles→Foot, shoulders→UpperArm, elbows→LowerArm,
  wrists→Hand, head→Head; spine1/2/3 spread over Spine/Chest/UpperChest;
  collars→Shoulder, neck→Neck and toes→Toes when present). Every mapped bone
  is driven — including neck and toes, whose rotations position their
  children.
- Bone orientations are reconstructed from the joint positions (port of the
  Blender retarget math, `MsflowRetarget.cs`) and written as `localRotation` in
  `LateUpdate`, blended by a fade weight, so playback smoothly leaves and
  returns to whatever the Animator is doing.
- **Rig rest pose is measured, not assumed** — T-Pose, A-Pose, or any custom
  stance all work without configuration. The `TransferMode` enum selects how
  the pose maps on:
  - `MatchSourcePose` (default): every bone aims exactly at the generated
    motion's joint directions — reproduces the generated motion faithfully on
    any rig rest. At the start of playback the character takes the motion's
    own first pose.
  - `RestAnchored`: the motion is applied as a rotation delta on the rig's
    captured stance, so playback starts exactly at the character's rest pose;
    limb aiming is approximate when the rig rest differs strongly from the
    generator's stance (arms-down canonical).
- Root motion modes: `Hips` (default; pelvis translation on the hips bone),
  `InPlace` (vertical bob only), `RootTransform` (moves the character root).
- Leg scale is matched from the hip→ankle span ratio.

Caveats:

- While playing, the generated pose **overrides** the Animator's output (it is
  not a muscle-space AnimationClip, so state-machine crossfading is not
  available — that is a possible follow-up).
- The rig's rest is measured when `Play` starts. With `MatchSourcePose` the
  pose follows the generated motion regardless; with `RestAnchored`, start
  playback while the character is in its normal stance.
- Twist around bones is under-determined by positions (same as the Blender
  bridge); expect slight candy-wrapper artifacts on extreme poses.
- Positions are metres, Y-up, facing +Z, character-right at +X. The generated
  data is X-mirrored relative to Unity (measured against the clips), and
  `MsflowSkeleton.FromDataConvention` flips X back on playback — the same flip
  the constraint builder applies on the way out, so authored pins round-trip
  exactly. If a motion still appears mirrored, compare against the run's
  `sample_00.mp4` and report it.

## 7. Tests

- Unity: open the package in a project and run
  `Window → General → Test Runner → EditMode` (parse/mapping/retarget and
  constraint-builder tests).
- Server: `uv run --group server python webui/test_runtime_server.py`
  (protocol selftest, no GPU needed).
- Numerical check of the retarget port outside Unity: the same round-trip
  property the Blender `retarget_selftest.py` asserts is covered by the
  EditMode tests.

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `not connected` / immediate error | Is `webui/runtime_server.py` running? `--bind 0.0.0.0` for NAT WSL2; firewall prompt may be needed |
| `unknown job_id` on fetch | Job cache expired (5 min) — request again |
| `n_frames … exceeds model max_length` | Lower the duration |
| `character is missing required humanoid bones: …` | The avatar lacks those `HumanBodyBones`; assign a full humanoid or relax the required set in `MsflowBoneMapper` |
| Character moves to the wrong place / drifts | Stop any external root motion during playback, or use `InPlace` |
| Motion looks mirrored vs `sample_00.mp4` | Report it — one-constant axis fix in the playback path |
