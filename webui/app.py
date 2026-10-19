"""MSFlow constraint-placement web UI (Gradio Blocks).

Run from the repo root (inside WSL2):
    export UV_PROJECT_ENVIRONMENT="$HOME/venvs/msflow"
    uv run --group ui python webui/app.py
Then open the printed URL from the Windows-side browser.

The editor authors sparse-control specs (the exact format consumed by
`sample/demo_joint_contorl.py`) by dragging joint constraints on a 3D
three.js armature over a HumanML3D reference pose, and launches the same
projection-sampling pipeline as the CLI.

Frame model: one "current frame" drives the armature preview, the underlay
and constraint editing. Canvas events are *client-authoritative* (selection
and drags never round-trip the scene value back); the server folds them into
the spec state and must respond to canvas events with `canvas=False`
refreshes to avoid feedback loops.
"""
from __future__ import annotations

import copy
import inspect
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

# isort: off
from webui.blender_bridge import DEFAULT_PORT as BLENDER_PORT  # noqa: E402
from webui.blender_bridge import push_motion, resolve_host_label  # noqa: E402
from webui.reference import ReferenceState, ReferenceUnavailable  # noqa: E402
from webui.runner import Runner, RunSettings  # noqa: E402
from webui.skeleton import (  # noqa: E402
    JOINT_CHAIN,
    JOINT_NAMES,
    NUM_JOINTS,
    canonical_pose,
    joint_label,
)
from webui.spec_schema import (  # noqa: E402
    AXES_BY_INDEX,
    STAGE_COLORS,
    Control,
    Spec,
    empty_spec,
    load_spec_file,
    parse_axes,
    safe_spec_name,
)

# isort: on

try:
    # via the `ui` dependency group (path dep; wheel built by `gradio cc build`)
    from gradio_constraint_canvas import ConstraintCanvas  # type: ignore
except Exception:  # pragma: no cover - only when dist is missing
    CANVAS_READY = False
    CANVAS_NOTE = (
        "⚠️ 3D scene editor unavailable (component not built). Install Node "
        "(`sudo apt install -y nodejs npm`), then run `gradio cc build` in "
        "webui/components/constraint_canvas. Numeric keyframe editing below "
        "still works."
    )
else:
    CANVAS_READY = True
    CANVAS_NOTE = ""

FRAMERATE = 20  # HumanML3D fps
SPECS_DIR = ROOT / "specs"
BUNDLED_SPECS = ROOT / "sample" / "joint_control_specs"
STAGES = ("early", "middle", "late")
WAYPOINT_COLOR = "#E6A83C"  # pelvis path pins (A → B), distinct from stages
WAYPOINT_JOINT = 0  # pelvis
WAYPOINT_AXES = (0, 2)  # x/z only: height stays free so the walk stays alive
MAX_VIDEO_SLOTS = 8
MAX_REF_CHOICES = 500
DEFAULT_JOINTS = (20, 21)  # left/right wrist — the paper's chair-drag example

WEBUI_BUILD = "b7"


def _log(message: str) -> None:
    caller = inspect.stack()[1].function
    stamp = time.strftime("%H:%M:%S")
    print(f"[webui {WEBUI_BUILD} {stamp} {caller}] {message}", flush=True)


runner = Runner()
reference = ReferenceState(ROOT / "datasets" / "HumanML3D")
_last_run: dict[str, str] = {"dir": ""}


# ---------------------------------------------------------------------------
# Editor state


def new_state() -> dict[str, Any]:
    """Fresh editor state with a starter spec on the right wrist."""
    state: dict[str, Any] = {
        "prompt": "a person drags a chair forward",
        "n_frames": 121,
        "source_motion_id": "",
        "frame": 0,
        "ref_len": 1,
        "sel_kf": 0,
        "sel_joint": DEFAULT_JOINTS[1],
        "controls": [],
    }
    pose = canonical_pose()
    for fraction in (0.0, 0.5, 1.0):
        frame = int(round(fraction * (state["n_frames"] - 1)))
        state["controls"].append(
            {
                "frame": frame,
                "frame_fraction": None,
                "joints": list(DEFAULT_JOINTS),
                "axes": [0, 2],
                "axes_raw": None,
                "values": [
                    [float(pose[j][0]), float(pose[j][1]), float(pose[j][2])]
                    for j in DEFAULT_JOINTS
                ],
                "stage": "middle",
                "role": "",
            }
        )
    return state


def _selected_control(state: dict[str, Any]) -> dict[str, Any] | None:
    index = state.get("sel_kf")
    index = -1 if index is None else int(index)  # note: 0 is a valid index
    controls = state["controls"]
    if 0 <= index < len(controls):
        return controls[index]
    return None


def _row_for(control: dict[str, Any] | None, joint: int) -> list[float] | None:
    if control is None or int(joint) not in control["joints"]:
        return None
    return control["values"][control["joints"].index(int(joint))]


def _selected_joint(state: dict[str, Any]) -> int | None:
    sel = state.get("sel_joint")
    return None if sel is None else int(sel)


def _widget_pose(state: dict[str, Any], frame: int | None) -> tuple[np.ndarray, str]:
    """Reference pose at an absolute frame; canonical fallback with a note."""
    ref_id = str(state.get("source_motion_id") or "")
    if ref_id:
        try:
            entry = reference.get(ref_id)
            index = int(frame if frame is not None else state["frame"])
            pose = np.asarray(entry["motion"])[index % entry["motion"].shape[0]]
            return pose.astype(np.float32), ""
        except ReferenceUnavailable as exc:
            return canonical_pose(), f"reference unavailable: {exc}"
    return canonical_pose(), ""


def _joint_row_from_reference(
    state: dict[str, Any], joint: int, frame: int
) -> list[float]:
    pose, _warn = _widget_pose(state, frame)
    return [float(pose[joint][0]), float(pose[joint][1]), float(pose[joint][2])]


def _new_control(
    state: dict[str, Any],
    frame: int,
    joints: list[int],
    stage: str = "middle",
    role: str = "",
) -> dict[str, Any]:
    return {
        "frame": int(frame),
        "frame_fraction": None,
        "joints": [int(j) for j in joints],
        "axes": [0, 2],
        "axes_raw": None,
        "values": [_joint_row_from_reference(state, j, frame) for j in joints],
        "stage": stage,
        "role": role,
    }


def _clamp_spec(state: dict[str, Any]) -> None:
    state["n_frames"] = int(state["n_frames"])
    state["frame"] = min(max(int(state.get("frame", 0)), 0), state["n_frames"] - 1)
    for control in state["controls"]:
        if not 0 <= control["frame"] < state["n_frames"]:
            control["frame"] = min(max(control["frame"], 0), state["n_frames"] - 1)
            control["frame_fraction"] = None
    if state["sel_kf"] >= len(state["controls"]):
        state["sel_kf"] = len(state["controls"]) - 1


def _common_status(state: dict[str, Any], warn: str = "", notice: str = "") -> str:
    bits: list[str] = []
    if CANVAS_READY:
        bits.append(
            "🖱 click a joint or constraint to select · drag the gizmo (axes "
            "auto-constrain) · drag the gizmo's SQUARE to slide along the "
            "ground · 🧭 waypoint = pelvis x/z pin, click it then drag · esc "
            "or click empty space deselects"
        )
    if CANVAS_NOTE:
        bits.append(CANVAS_NOTE)
    if state.get("warn"):
        bits.append(f"⚠️ {state['warn']}")
    if warn:
        bits.append(f"⚠️ {warn}")
    if notice:
        bits.append(notice)
    control = _selected_control(state)
    if control is not None:
        names = ", ".join(JOINT_NAMES.get(j, f"j{j}") for j in control["joints"])
        if control.get("role") == "waypoint":
            bits.append(
                f"editing WAYPOINT (pelvis x/z, height free) at frame "
                f"{control['frame']} — the torso and legs stay animated"
            )
        else:
            bits.append(
                f"editing keyframe #{state['sel_kf'] + 1} · frame {control['frame']}"
                + (f" · joints {names}" if names else "")
            )
    else:
        bits.append(
            f"frame {int(state['frame'])} has no keyframe — drag a joint or use "
            "“constrain this frame”."
        )
    return "\n".join(bits)


# ---------------------------------------------------------------------------
# Canvas / preview payloads


def canvas_value(state: dict[str, Any]) -> dict[str, Any]:
    """Scene payload for the canvas: every saved constraint is a handle
    (id + frame + constrained axes). Handles at the current frame render solid
    and gizmo-able; the rest render as ghosts. The underlay is the base pose
    at the current frame; the client IK-solves the armature onto the handles."""
    frame = int(state["frame"])
    control = _selected_control(state)
    underlay, _warn = _widget_pose(state, frame)
    points = [
        {"joint": j, "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
        for j, p in enumerate(underlay)
    ]
    sel_joint = _selected_joint(state)
    sel_flag = -1 if sel_joint is None else sel_joint  # note: 0 is a real joint
    markers_by_id: dict[str, dict[str, Any]] = {}
    for candidate in state["controls"]:
        c_frame = int(candidate["frame"])
        role = str(candidate.get("role") or "")
        color = (
            WAYPOINT_COLOR
            if role == "waypoint"
            else STAGE_COLORS.get(candidate["stage"] or "middle", "#CC79A7")
        )
        axes_names = [
            AXES_BY_INDEX[a] for a in (candidate["axes"] or [0, 1, 2]) if 0 <= a < 3
        ]
        is_current = c_frame == frame
        for row_i, joint in enumerate(candidate["joints"]):
            row = candidate["values"][row_i]
            marker_id = f"f{c_frame}-j{int(joint)}"
            # later entries win, mirroring load_sparse_raw_control_spec overwrite
            markers_by_id[marker_id] = {
                "id": marker_id,
                "joint": int(joint),
                "frame": c_frame,
                "x": float(row[0]),
                "y": float(row[1]),
                "z": float(row[2]),
                "axes": axes_names,
                "color": color,
                "controlled": True,
                "waypoint": role == "waypoint",
                "selected": bool(is_current and int(joint) == sel_flag),
            }
    markers = list(markers_by_id.values())

    lines: list[dict[str, Any]] = []
    # ordered pelvis waypoints draw the A → B path the motion should trace
    waypoint_track: list[list[float]] = []
    for candidate in sorted(
        (c for c in state["controls"] if c.get("role") == "waypoint"),
        key=lambda c: int(c["frame"]),
    ):
        for row in candidate["values"]:
            waypoint_track.append([float(row[0]), float(row[1]), float(row[2])])
    if len(waypoint_track) >= 2:
        lines.append({"points": waypoint_track, "color": WAYPOINT_COLOR})
    if control is not None:
        for joint in control["joints"]:
            track: list[list[float]] = []
            for other in state["controls"]:
                row = _row_for(other, int(joint))
                if row is not None:
                    track.append([float(row[0]), float(row[1]), float(row[2])])
            if len(track) >= 2:
                lines.append({"points": track, "color": STAGE_COLORS["middle"]})

    return {
        "frame": frame,
        "underlay": {"points": points, "edges": [list(ele) for ele in JOINT_CHAIN]},
        "markers": markers,
        "lines": lines,
        "selected_joint": _selected_joint(state),
        "label": (
            f"frame {frame}/{int(state['n_frames']) - 1} · "
            f"{len(state['controls'])} constraint groups"
        ),
        "playback": state.get("playback") or None,
    }


def _kf_choices(state: dict[str, Any]) -> list[str]:
    return [
        f"#{i + 1} · {'waypoint' if c.get('role') == 'waypoint' else 'kf'} @ "
        f"frame {c['frame']} ({c['frame'] / FRAMERATE:.2f}s)"
        for i, c in enumerate(state["controls"])
    ]


def _kf_label(state: dict[str, Any], index: int) -> str | None:
    if 0 <= index < len(state["controls"]):
        return _kf_choices(state)[index]
    return None


def _kf_index_from_label(label: str | None) -> int:
    if not label:
        return -1
    try:
        return int(label.split("·")[0].lstrip("# ")) - 1
    except ValueError:
        return -1


def _resolve_sel_kf(state: dict[str, Any], label: str | None = None) -> int:
    """Resolve the keyframe selection, with the dropdown's live value winning.

    The dropdown's change event and the panel/button events run on separate
    gradio listeners and can race; reading the label from each event's own
    input makes the mutators immune to a stale state['sel_kf'].
    """
    index = _kf_index_from_label(label)
    if not 0 <= index < len(state["controls"]):
        current = state.get("sel_kf")
        index = -1 if current is None else int(current)
    if 0 <= index < len(state["controls"]):
        state["sel_kf"] = index
    return index


def _ref_options() -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = [("— none (canonical pose) —", "")]
    for stem in reference.list_ids()[:MAX_REF_CHOICES]:
        options.append((f"motion {stem}", stem))
    return options


def _spec_options() -> list[tuple[str, str]]:
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    options: list[tuple[str, str]] = []
    for label, folder in (("bundled", BUNDLED_SPECS), ("user", SPECS_DIR)):
        if folder.is_dir():
            for path in sorted(folder.glob("*.json")):
                options.append((f"{label}: {path.stem}", str(path)))
    return options


def spec_from_state(state: dict[str, Any]) -> Spec:
    spec = empty_spec(n_frames=int(state["n_frames"]), prompt=state["prompt"])
    spec.source_motion_id = str(state.get("source_motion_id") or "")
    for control in state["controls"]:
        spec.controls.append(
            Control(
                frame=int(control["frame"]),
                joints=[int(j) for j in control["joints"]],
                axes=[int(a) for a in control["axes"]],
                values=[[float(v) for v in row] for row in control["values"]],
                stage=control.get("stage", "middle"),
                role=control.get("role", ""),
                frame_fraction=control.get("frame_fraction"),
                axes_raw=control.get("axes_raw"),
            )
        )
    if not spec.controls:
        raise ValueError("Add at least one keyframe before saving/running.")
    for index, control in enumerate(spec.controls):
        if not control.joints:
            raise ValueError(f"Keyframe #{index + 1} has no joints — add or delete it.")
        if len(set(control.joints)) != len(control.joints):
            raise ValueError(f"Keyframe #{index + 1} lists duplicate joints.")
    return spec


# ---------------------------------------------------------------------------
# Refresh contract. Every editor event handler returns exactly this tuple:
# (state, canvas, kf_dropdown, joints_cbg, axes_cbg, kf_frame, kf_stage,
#  x_val, y_val, z_val, prompt, n_frames, ref_id_dd, frame, status)

EDIT_OUTPUTS = (
    "canvas",
    "kf_dropdown",
    "joints_cbg",
    "axes_cbg",
    "kf_frame_slider",
    "kf_stage",
    "x_val",
    "y_val",
    "z_val",
    "prompt_box",
    "n_frames_slider",
    "duration_box",
    "ref_id_dd",
    "frame_slider",
    "status",
)


def refresh(
    state: dict[str, Any],
    warn: str = "",
    notice: str = "",
    light: bool = False,
    kf_frame_bounds: bool = False,
    canvas: bool = True,
) -> tuple[Any, ...]:
    """Re-emit the editor widgets (state first, then EDIT_OUTPUTS order).

    light=True: only the state really updates. Used for high-frequency events
    (prompt keystrokes) and for programmatic canvas value changes (whose
    canvas.change handler would otherwise re-issue another canvas value and
    re-trigger itself forever).
    kf_frame_bounds=True: light, plus both frame sliders' bounds and the
    duration field (an n_frames change moves all three).
    canvas=False: full refresh that leaves the canvas slot as a no-op. This
    is REQUIRED for handlers invoked *by* canvas events (select/move): the
    canvas is client-authoritative during interaction, and re-issuing the
    canvas value in response to a canvas event would clobber the live scene
    and selection.
    """
    noop: Any = gr.update()
    caller = inspect.stack()[1].function
    if light:
        slots: list[Any] = [noop] * len(EDIT_OUTPUTS)
        if kf_frame_bounds:
            slots[EDIT_OUTPUTS.index("kf_frame_slider")] = gr.update(
                minimum=0, maximum=state["n_frames"] - 1, step=1
            )
            slots[EDIT_OUTPUTS.index("frame_slider")] = gr.update(
                minimum=0, maximum=state["n_frames"] - 1, step=1
            )
            slots[EDIT_OUTPUTS.index("duration_box")] = gr.update(
                value=round(state["n_frames"] / FRAMERATE, 2)
            )
        _log(f"from {caller}: light refresh (canvas untouched)")
        return (state, *slots)
    _clamp_spec(state)
    control = _selected_control(state)
    sel_joint = _selected_joint(state)
    _pose, ref_warn = _widget_pose(state, state["frame"])
    status = _common_status(state, warn or ref_warn, notice)

    x_value = y_value = z_value = None
    if control is not None and sel_joint is not None:
        row = _row_for(control, sel_joint)
        if row is not None:
            x_value, y_value, z_value = float(row[0]), float(row[1]), float(row[2])
    axes_value = [AXES_BY_INDEX[a] for a in (control["axes"] if control else [0, 2])]
    joints_value = [joint_label(j) for j in (control["joints"] if control else [])]

    updates = {
        "canvas": canvas_value(state) if CANVAS_READY else noop,
        "kf_dropdown": gr.update(
            choices=_kf_choices(state), value=_kf_label(state, state["sel_kf"])
        ),
        "joints_cbg": gr.update(
            choices=[joint_label(j) for j in range(NUM_JOINTS)], value=joints_value
        ),
        "axes_cbg": gr.update(choices=["x", "y", "z"], value=axes_value),
        "kf_frame_slider": gr.update(
            minimum=0,
            maximum=state["n_frames"] - 1,
            step=1,
            value=(control["frame"] if control else int(state["frame"])),
        ),
        "kf_stage": gr.update(
            choices=list(STAGES),
            value=(control["stage"] or "middle") if control else "middle",
        ),
        "x_val": gr.update(value=x_value),
        "y_val": gr.update(value=y_value),
        "z_val": gr.update(value=z_value),
        "prompt_box": gr.update(value=state["prompt"]),
        "n_frames_slider": gr.update(
            minimum=16, maximum=1024, step=1, value=state["n_frames"]
        ),
        "duration_box": gr.update(value=round(state["n_frames"] / FRAMERATE, 2)),
        "ref_id_dd": gr.update(
            choices=_ref_options(), value=state.get("source_motion_id") or None
        ),
        "frame_slider": gr.update(
            minimum=0,
            maximum=state["n_frames"] - 1,
            step=1,
            value=int(state["frame"]),
        ),
        "status": gr.update(value=status),
    }
    if not canvas:
        # canvas-originated event: never re-issue the canvas value
        updates["canvas"] = noop
    _log(
        f"from {caller}: "
        f"canvas={'RE-ISSUE' if updates['canvas'] is not noop else 'no'} "
        f"sel_kf={state.get('sel_kf')} frame={state.get('frame')} "
        f"sel_joint={state.get('sel_joint')}"
    )
    return (state, *(updates[name] for name in EDIT_OUTPUTS))


# ---------------------------------------------------------------------------
# Event handlers


def on_ref_changed(state: dict[str, Any], ref_id: str) -> tuple[Any, ...]:
    state["source_motion_id"] = ref_id or ""
    if ref_id:
        try:
            state["ref_len"] = int(reference.get(ref_id)["motion"].shape[0])
            state["warn"] = ""
        except Exception as exc:  # degrade to the canonical underlay, never fail
            state["ref_len"] = 1
            state[
                "warn"
            ] = f"reference '{ref_id}' unusable ({exc}); using canonical pose"
    state["frame"] = 0
    return refresh(state)


def _kf_indices_at(
    state: dict[str, Any], frame: int, joint: int | None = None
) -> list[int]:
    """Indices of keyframes at `frame`, optionally filtered by membership."""
    return [
        index
        for index, control in enumerate(state["controls"])
        if int(control["frame"]) == int(frame)
        and (joint is None or int(joint) in control["joints"])
    ]


def on_frame_changed(state: dict[str, Any], value: int) -> tuple[Any, ...]:
    """Single current-frame slider: the scene follows. The keyframe selection
    STICKS while scrubbing — the selected keyframe is the retime/snapshot
    target; use the dropdown to change the selection."""
    if int(value) == int(state["frame"]):
        # programmatic echo of our own panel update — keep the live scene
        return refresh(state, light=True)
    state["frame"] = int(value)
    _clamp_spec(state)
    return refresh(state)


def on_canvas_change(
    payload: dict[str, Any],
    state: dict[str, Any],
    kf_label: str | None = None,
) -> tuple[Any, ...]:
    """Handle canvas events.

    Payload is the dispatch detail ({action: move|select|snapshot, ...}) for
    user interactions, or the component's own value for programmatic value
    changes (no 'action' key) — the latter must return a *light* refresh:
    re-issuing a canvas value here would fire canvas.change again, forever.

    The canvas is client-authoritative during interaction: selection and drag
    commits only fold state into the server (`canvas=False` refresh so the
    numeric panel follows) and never re-issue the canvas value. Only genuine
    context switches (ghost jump, snapshot) return a fresh canvas value.
    """
    payload = payload or {}
    if isinstance(payload, dict) and "action" not in payload:
        # some gradio transports wrap the dispatch detail as {"value": {...}};
        # unwrap so the action still reaches the handler
        inner = payload.get("value")
        if isinstance(inner, dict) and "action" in inner:
            _log(f"unwrapped value-wrapped payload with action={inner.get('action')}")
            payload = inner
    action = payload.get("action") if isinstance(payload, dict) else None
    if action is None:
        keys = (
            sorted(payload)[:8] if isinstance(payload, dict) else type(payload).__name__
        )
        _log(f"value echo (no action) keys={keys}")
        return refresh(state, light=True)
    brief = {
        key: value
        for key, value in payload.items()
        if key not in ("pose", "underlay", "markers", "lines", "playback")
    }
    if "pose" in payload:
        brief["pose_rows"] = len(payload["pose"])
    _log(f"event action={action} {brief}")
    if action == "select":
        joint = int(payload.get("joint", -1))
        frame = payload.get("frame")
        state["sel_joint"] = None if joint < 0 else joint
        if joint < 0:
            return refresh(state, canvas=False)
        if frame is not None and int(frame) != int(state["frame"]):
            # ghost handle of another frame: jump the editing frame there and
            # select the keyframe that owns this constraint
            state["frame"] = int(frame)
            _clamp_spec(state)
            matches = _kf_indices_at(state, state["frame"], joint)
            state["sel_kf"] = matches[-1] if matches else -1
            return refresh(state)
        return refresh(state, canvas=False)
    if action == "move":
        joint = int(payload.get("joint", -1))
        if joint < 0:
            return refresh(state, light=True)
        client_frame = payload.get("frame")
        cur_frame = (
            int(client_frame) if client_frame is not None else int(state["frame"])
        )
        if client_frame is not None and int(state["frame"]) != cur_frame:
            _log(
                f"frame mismatch: state={state['frame']} client={cur_frame} "
                "-> adopting client frame"
            )
            state["frame"] = cur_frame
            _clamp_spec(state)
        control = _selected_control(state)
        if control is None or int(control["frame"]) != cur_frame:
            # edit at the current frame: reuse a keyframe there (prefer one that
            # holds this joint), else create one — never land in a stale keyframe
            matches = _kf_indices_at(state, cur_frame, joint)
            if not matches:
                matches = _kf_indices_at(state, cur_frame)
            if matches:
                state["sel_kf"] = matches[-1]
            else:
                state["controls"].append(_new_control(state, cur_frame, [joint]))
                state["sel_kf"] = len(state["controls"]) - 1
            control = _selected_control(state)
        state["sel_joint"] = joint
        row = _row_for(control, joint)
        if row is None and control is not None:
            control["joints"].append(int(joint))
            control["values"].append(
                _joint_row_from_reference(state, int(joint), int(control["frame"]))
            )
            row = _row_for(control, joint)
        axis_map = {"x": 0, "y": 1, "z": 2}
        touched = [
            name
            for name in (payload.get("axes") or ["x", "y", "z"])
            if name in axis_map
        ]
        if not touched:
            touched = ["x", "y", "z"]
        if row is not None and control is not None:
            # write the FULL dropped position (not only the touched axes):
            # untouched axes of a raw-joint drag come from the live solved pose,
            # not the reference-seeded row — otherwise the next canvas sync
            # snaps the handle back (the "moved position resets" bug)
            for name in ("x", "y", "z"):
                if payload.get(name) is not None:
                    row[axis_map[name]] = float(payload[name])
            for name in touched:
                index = axis_map[name]
                if index not in control["axes"]:
                    control["axes"].append(index)
            control["axes"].sort()
            control["axes_raw"] = None
            _log(
                f"fold move: joint={joint} -> kf#{state['sel_kf']} "
                f"frame={control['frame']} row={row} axes={control['axes']}"
            )
            return refresh(state, canvas=False)
        return refresh(state, canvas=False)
    if action == "snapshot":
        try:
            _resolve_sel_kf(state, kf_label)
            control = _selected_control(state)
            if control is not None:
                # pin into the SELECTED keyframe (possibly a past frame) —
                # never silently grow the spec with a new next-frame keyframe
                frame = int(control["frame"])
            else:
                frame = int(payload.get("frame", state["frame"]))
            notice = _snapshot_frame(state, frame, payload.get("pose") or [])
        except Exception as exc:
            return refresh(state, f"snapshot ignored: {exc}")
        return refresh(state, notice=notice)
    return refresh(state, light=True)


def _snapshot_frame(state: dict[str, Any], frame: int, pose_rows: list[Any]) -> str:
    """Replace every constraint at `frame` with a full-pose constraint
    (all 22 joints, x/y/z constrained) captured from the solved preview."""
    _clamp_spec(state)
    frame = int(frame)
    _log(f"snapshot fold: frame={frame} pose_rows={len(pose_rows)}")
    if len(pose_rows) != NUM_JOINTS:
        raise ValueError(f"expected {NUM_JOINTS} joint rows, got {len(pose_rows)}")
    values = [[float(row[0]), float(row[1]), float(row[2])] for row in pose_rows]
    stage = "middle"
    first_index: int | None = None
    kept: list[dict[str, Any]] = []
    for index, candidate in enumerate(state["controls"]):
        if int(candidate["frame"]) == frame:
            if first_index is None:
                first_index = index
                stage = candidate.get("stage") or "middle"
            continue  # replaced
        kept.append(candidate)
    pinned: dict[str, Any] = {
        "frame": frame,
        "frame_fraction": None,
        "joints": list(range(NUM_JOINTS)),
        "axes": [0, 1, 2],
        "axes_raw": None,
        "values": values,
        "stage": stage,
        "role": "",
    }
    if first_index is None:
        kept.append(pinned)
        first_index = len(kept) - 1
    else:
        kept.insert(first_index, pinned)
    state["controls"] = kept
    state["sel_kf"] = first_index
    state["sel_joint"] = None
    return f"pinned full pose ({NUM_JOINTS} joints) at frame {frame}"


def on_kf_select(state: dict[str, Any], label: str | None) -> tuple[Any, ...]:
    if not label:
        index = -1
    else:
        try:
            index = int(label.split("·")[0].lstrip("# ")) - 1
        except ValueError:
            index = -1
    current = state.get("sel_kf")
    current = -1 if current is None else int(current)
    if index == current:
        # programmatic echo of our own panel update — never rebuild the scene
        return refresh(state, light=True)
    state["sel_kf"] = index
    control = _selected_control(state)
    if control is not None:
        # jump the scene to the selected keyframe's frame
        state["frame"] = int(control["frame"])
    return refresh(state)


def on_add_keyframe(state: dict[str, Any], frame: int) -> tuple[Any, ...]:
    joints = [_selected_joint(state) or DEFAULT_JOINTS[0]]
    frame = min(max(int(frame), 0), int(state["n_frames"]) - 1)
    state["controls"].append(_new_control(state, frame, joints))
    state["sel_kf"] = len(state["controls"]) - 1
    state["sel_joint"] = joints[0]
    state["frame"] = frame  # the scene follows the new keyframe
    return refresh(state, notice=f"added keyframe at frame {frame}")


def on_add_waypoint(state: dict[str, Any], frame: int) -> tuple[Any, ...]:
    """Pelvis path waypoint: pins joint 0's x/z at the current frame so the
    root passes through the point (A → B → …) while height, torso and legs
    stay animated — unlike locking a spine joint to a position."""
    _clamp_spec(state)
    frame = min(max(int(frame), 0), int(state["n_frames"]) - 1)
    dedicated = [
        index
        for index, candidate in enumerate(state["controls"])
        if int(candidate["frame"]) == frame and candidate.get("role") == "waypoint"
    ]
    if dedicated:
        # repeated clicks keep the placed spot; only follow the marker
        state["sel_kf"] = dedicated[0]
        notice = f"waypoint already at frame {frame} — drag it or edit x/z below"
    else:
        absorbed = [
            index
            for index, candidate in enumerate(state["controls"])
            if int(candidate["frame"]) == frame and WAYPOINT_JOINT in candidate["joints"]
        ]
        if absorbed:
            # a shared keyframe already pins the pelvis here; reuse, don't clobber
            shared = state["controls"][absorbed[0]]
            shared["axes"] = sorted(set(shared["axes"]) | set(WAYPOINT_AXES))
            state["sel_kf"] = absorbed[0]
            notice = (
                f"pelvis constraint exists at frame {frame} — keyframe reused "
                "(unlock y here if you only want the path pin)"
            )
        else:
            control = _new_control(
                state, frame, [WAYPOINT_JOINT], role="waypoint"
            )
            control["axes"] = list(WAYPOINT_AXES)
            state["controls"].append(control)
            state["sel_kf"] = len(state["controls"]) - 1
            notice = f"waypoint added at frame {frame} (pelvis x/z pin)"
    state["sel_joint"] = WAYPOINT_JOINT
    state["frame"] = frame
    return refresh(state, notice=notice)


def on_duplicate_keyframe(
    state: dict[str, Any], kf_label: str | None = None
) -> tuple[Any, ...]:
    _resolve_sel_kf(state, kf_label)
    control = _selected_control(state)
    if control is None:
        return refresh(state, "Nothing to duplicate.")
    state["controls"].insert(state["sel_kf"] + 1, copy.deepcopy(control))
    state["sel_kf"] += 1
    return refresh(state)


def on_delete_keyframe(
    state: dict[str, Any], kf_label: str | None = None
) -> tuple[Any, ...]:
    _resolve_sel_kf(state, kf_label)
    index = int(state["sel_kf"])
    if 0 <= index < len(state["controls"]):
        state["controls"].pop(index)
        state["sel_kf"] = index - 1 if index >= len(state["controls"]) else index
        state["sel_kf"] = max(state["sel_kf"], -1)
    return refresh(state)


def on_joints_changed(
    state: dict[str, Any], labels: list[str], kf_label: str | None = None
) -> tuple[Any, ...]:
    _resolve_sel_kf(state, kf_label)
    wanted = sorted(
        dict.fromkeys(
            int(token)
            for item in labels or []
            for token in [item.split(" ", 1)[0]]
            if token.isdigit()
        )
    )
    control = _selected_control(state)
    if control is not None and sorted(control["joints"]) == wanted:
        # echo of our own panel update — keep the live scene
        return refresh(state, light=True)
    if control is None:
        if not wanted:
            return refresh(state)
        state["controls"].append(_new_control(state, int(state["frame"]), wanted))
        state["sel_kf"] = len(state["controls"]) - 1
        return refresh(state)
    kept_joints: list[int] = []
    kept_rows: list[list[float]] = []
    for joint, row in zip(control["joints"], control["values"]):
        if joint in wanted:
            kept_joints.append(int(joint))
            kept_rows.append(row)
    for joint in wanted:
        if joint not in kept_joints:
            kept_joints.append(int(joint))
            kept_rows.append(
                _joint_row_from_reference(state, joint, int(control["frame"]))
            )
    control["joints"], control["values"] = kept_joints, kept_rows
    if _selected_joint(state) is not None and _selected_joint(state) not in kept_joints:
        state["sel_joint"] = None
    return refresh(state)


def on_axes_changed(
    state: dict[str, Any], labels: list[str], kf_label: str | None = None
) -> tuple[Any, ...]:
    _resolve_sel_kf(state, kf_label)
    control = _selected_control(state)
    if control is None:
        return refresh(state, "Select a keyframe first.")
    parsed = parse_axes(",".join(labels)) if labels else []
    if parsed:
        if sorted(parsed) == sorted(control["axes"]):
            # echo of our own panel update — keep the live scene
            return refresh(state, light=True)
        control["axes"] = [int(a) for a in parsed]
        control["axes_raw"] = None
        return refresh(state)
    return refresh(state, "Keep at least one axis constrained (ignored).")


def on_kf_frame_changed(
    state: dict[str, Any], value: int, kf_label: str | None = None
) -> tuple[Any, ...]:
    _resolve_sel_kf(state, kf_label)
    control = _selected_control(state)
    if control is None:
        return refresh(state)
    control["frame"] = int(value)
    control["frame_fraction"] = None
    state["frame"] = int(value)  # the scene follows the retimed keyframe
    return refresh(state)


def on_stage_changed(
    state: dict[str, Any], value: str, kf_label: str | None = None
) -> tuple[Any, ...]:
    _resolve_sel_kf(state, kf_label)
    control = _selected_control(state)
    if control is None:
        return refresh(state)
    if str(control.get("stage") or "middle") == str(value or "middle"):
        # echo of our own panel update — keep the live scene
        return refresh(state, light=True)
    control["stage"] = value
    return refresh(state)


def on_value_changed(
    state: dict[str, Any],
    axis_index: int,
    value: float,
    kf_label: str | None = None,
) -> tuple[Any, ...]:
    _resolve_sel_kf(state, kf_label)
    joint = _selected_joint(state)
    control = _selected_control(state)
    if control is None or joint is None:
        # echo after a deselect — never rebuild the scene from this
        return refresh(state, light=True)
    row = _row_for(control, joint)
    if row is not None:
        new_value = float(value or 0.0)
        # the panel echoes back display-rounded values (Number precision=4,
        # slider step=0.01) of float32 rows — compare at widget precision,
        # otherwise every response re-fires these handlers and rebuilds
        tolerance = 5e-3 if axis_index == 1 else 5e-5
        if abs(row[axis_index] - new_value) <= tolerance:
            # programmatic echo of our own panel update — nothing to do
            return refresh(state, light=True)
        row[axis_index] = new_value
    return refresh(state)


def on_x_changed(
    state: dict[str, Any], value: float, kf_label: str | None = None
) -> tuple[Any, ...]:
    return on_value_changed(state, 0, value, kf_label)


def on_y_changed(
    state: dict[str, Any], value: float, kf_label: str | None = None
) -> tuple[Any, ...]:
    return on_value_changed(state, 1, value, kf_label)


def on_z_changed(
    state: dict[str, Any], value: float, kf_label: str | None = None
) -> tuple[Any, ...]:
    return on_value_changed(state, 2, value, kf_label)


def on_prompt_changed(state: dict[str, Any], value: str) -> tuple[Any, ...]:
    state["prompt"] = value
    return refresh(state, light=True)  # never rebuild the scene on keystrokes


def on_n_frames_changed(state: dict[str, Any], value: int) -> tuple[Any, ...]:
    state["n_frames"] = int(value)
    _clamp_spec(state)
    return refresh(state)  # full: clamped frames must reach the canvas ghosts


def on_duration_changed(state: dict[str, Any], value: float) -> tuple[Any, ...]:
    seconds = max(0.05, float(value or 0.05))
    new_frames = int(min(1024, max(16, round(seconds * FRAMERATE))))
    if new_frames == int(state["n_frames"]):
        return refresh(state, light=True)  # programmatic echo, nothing changed
    state["n_frames"] = new_frames
    _clamp_spec(state)
    return refresh(state)


def on_load_spec(state: dict[str, Any], path: str) -> tuple[Any, ...]:
    if not path:
        return refresh(state)
    try:
        spec = load_spec_file(path)
    except Exception as exc:
        return refresh(state, f"could not load spec: {exc}")
    state["prompt"] = spec.prompt
    state["n_frames"] = spec.n_frames
    state["source_motion_id"] = spec.source_motion_id
    state["controls"] = [
        {
            "frame": control.frame,
            "frame_fraction": control.frame_fraction,
            "joints": list(control.joints),
            "axes": list(control.axes),
            "axes_raw": control.axes_raw,
            "values": [[float(v) for v in row] for row in control.values],
            "stage": control.stage,
        }
        for control in spec.controls
    ]
    state["sel_kf"] = 0 if state["controls"] else -1
    state["sel_joint"] = None
    first = _selected_control(state)
    state["frame"] = int(first["frame"]) if first is not None else 0
    notice = f"loaded {path}"
    if spec.source_motion_id:
        try:
            state["ref_len"] = int(
                reference.get(spec.source_motion_id)["motion"].shape[0]
            )
        except ReferenceUnavailable as exc:
            state["warn"] = str(exc)
    _clamp_spec(state)
    return refresh(state, notice=notice)


def on_refresh_specs(state: dict[str, Any]) -> tuple[Any, ...]:
    state["warn"] = ""
    return refresh(state, notice="spec list refreshed")


def on_save_spec(state: dict[str, Any], name: str) -> tuple[Any, ...]:
    try:
        spec = spec_from_state(state)
    except Exception as exc:
        return refresh(state, f"spec unusable: {exc}")
    stem = safe_spec_name(name or "")
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPECS_DIR / f"{stem}.json"
    spec.dump(path)
    return refresh(state, notice=f"saved {path}")


def _checkpoint_options() -> tuple[list[str], list[str], str]:
    entries = runner.list_checkpoints()
    names = [entry["name"] for entry in entries]
    default = (
        "MMDiT_xyz_pretrained"
        if "MMDiT_xyz_pretrained" in names
        else (names[0] if names else "")
    )
    ckpts = next(
        (entry["ckpts"] for entry in entries if entry["name"] == default), ["latest"]
    )
    return names, ckpts, default


def on_checkpoint_change(
    state: dict[str, Any], checkpoint: str | None
) -> tuple[Any, ...]:
    entries = runner.list_checkpoints()
    ckpts = next(
        (entry["ckpts"] for entry in entries if entry["name"] == (checkpoint or "")),
        ["latest"],
    )
    return (state, gr.update(choices=ckpts, value=(ckpts[0] if ckpts else None)))


def on_refresh_checkpoints(state: dict[str, Any]) -> tuple[Any, ...]:
    names, ckpts, default = _checkpoint_options()
    return (
        state,
        gr.update(choices=names, value=default),
        gr.update(choices=ckpts, value=(ckpts[0] if ckpts else None)),
    )


# ---------------------------------------------------------------------------
# Generation. Outputs: (state, status, out_dir_md, zip_file, *video_slots(8))


def on_run(
    state: dict[str, Any],
    checkpoint_name: str | None,
    checkpoint_ckpt: str | None,
    device: str,
    spec_name: str,
    num_samples: int,
    num_steps: int,
    cfg: float,
    w_kin: float,
    seed: float,
    use_metric_R: bool,
    use_augmented_obs: bool,
    use_noise_mixing: bool,
    noise_mix_strength: float,
    save_mp4: bool,
    save_gif: bool,
    progress: gr.Progress = gr.Progress(),  # noqa: B008
) -> tuple[Any, ...]:
    hidden = [gr.update(visible=False)] * MAX_VIDEO_SLOTS
    neutral = (gr.update(), gr.update(), gr.update())  # out_md, zip, canvas
    try:
        spec = spec_from_state(state)
    except Exception as exc:
        return (state, f"⛔ spec unusable: {exc}", *neutral, *hidden)
    payload = spec.to_json_dict()
    payload["spec_name"] = safe_spec_name(spec_name) or "run"
    settings = RunSettings(
        name=checkpoint_name or "MMDiT_xyz_pretrained",
        ckpt=checkpoint_ckpt or "latest",
        device=str(device).split(" ")[0],
        num_samples=int(num_samples),
        num_steps=int(num_steps),
        cfg=float(cfg),
        w_kin=float(w_kin),
        seed=int(seed),
        use_metric_R=bool(use_metric_R),
        use_augmented_obs=bool(use_augmented_obs),
        use_noise_mixing=bool(use_noise_mixing),
        noise_mix_strength=(
            float(noise_mix_strength) if noise_mix_strength is not None else None
        ),
        save_mp4=bool(save_mp4),
        save_gif=bool(save_gif),
    )
    if runner.is_busy():
        return (state, "⛔ a generation is already running", *neutral, *hidden)
    try:
        result = runner.run(payload, settings, _progress_bridge(progress))
    except Exception as exc:
        return (state, f"⛔ generation failed: {exc}", *neutral, *hidden)
    if getattr(result, "motion", None):
        state["playback"] = {
            "frames": result.motion,
            "fps": FRAMERATE,
            "label": "result sample 00",
        }
    zip_path = ""
    try:
        zip_path = shutil.make_archive(
            str(Path(result.out_dir) / "bundle"), "zip", result.out_dir
        )
    except Exception:
        zip_path = ""
    _last_run["dir"] = str(result.out_dir)
    status = (
        f"✅ wrote {result.out_dir}\n"
        f"seed {settings.seed} · cfg {settings.cfg} · steps {settings.num_steps} · "
        f"w_kin {settings.w_kin}"
    )
    out_md = gr.update(value=result.out_dir, visible=True)
    bundles = gr.update(value=zip_path or None, visible=bool(zip_path))
    shown = result.videos[:MAX_VIDEO_SLOTS]
    shown = list(shown)
    slots = [gr.update(value=value, visible=True) for value in shown]
    slots += [gr.update(visible=False)] * (MAX_VIDEO_SLOTS - len(shown))
    return (state, status, out_md, bundles, canvas_value(state), *slots)


def on_send_to_blender(
    run_dir: Any, host: str, port: Any
) -> str:  # outputs: status
    """Push the latest run's exported .msflow.json files to the Blender
    bridge add-on (filename-based; the add-on imports over the shared drive
    or from the same POST body's path)."""
    folder = (str(run_dir or "").strip() or _last_run.get("dir", "")).strip()
    if not folder or not Path(folder).is_dir():
        return "📡 send to Blender: no run directory yet — generate first."
    files = sorted(Path(folder).glob("sample_*.msflow.json"))
    if not files:
        return (
            f"📡 send to Blender: no sample_*.msflow.json in {folder} — runs made "
            "before the export hook need `python -m webui.msflow_export --run "
            f"{folder}`."
        )
    target = (host or "auto").strip() or "auto"
    port_value = int(port) if port is not None else BLENDER_PORT
    lines = [f"📡 send to Blender → {resolve_host_label(target)}:{port_value}"]
    failed = 0
    for path in files:
        ok, message = push_motion(path, host=target, port=port_value)
        failed += 0 if ok else 1
        lines.append(f"{'✅' if ok else '⛔'} {path.name}: {message}")
    if failed:
        lines.append(
            "hint: start the receiver in Blender's N panel → MSFlow; under WSL2 "
            "NAT also enable 'bind all' there and allow the firewall prompt, or "
            "use the add-on's Import latest / watch folder."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def _progress_bridge(progress: gr.Progress) -> Any:
    def cb(fraction: float, message: str) -> None:
        try:
            progress(float(min(max(fraction, 0.0), 1.0)), desc=message)
        except Exception:
            pass

    return cb


# ---------------------------------------------------------------------------
# UI construction


def gradio_placeholder() -> str:
    return (
        "<div style='padding:40px 20px;border:1px dashed #b6b6c8;border-radius:10px;"
        "color:#555;text-align:center;line-height:1.6;font-family:sans-serif'>"
        "🧩 3D scene placeholder<br/><small>build the ConstraintCanvas "
        "component (Node toolchain) to enable the scene editor</small></div>"
    )


def build_ui() -> gr.Blocks:
    names, ckpts, default_name = _checkpoint_options()
    starter = new_state()
    devices = runner.list_devices()
    with gr.Blocks(title="MSFlow constraint editor") as demo:
        gr.Markdown(
            "### 🖌️ MSFlow constraint editor\n"
            "Drag joint constraints on the 3D armature (a live IK estimate "
            "poses the skeleton toward them), pin whole poses with “constrain "
            "this frame”, tune stage/axes, run the sampler.\n"
            "Specs are round-trip-safe with `sample/demo_joint_contorl.py`."
        )
        with gr.Row():
            with gr.Column(scale=6):
                canvas_el = (
                    ConstraintCanvas(label="3D constraint scene editor")
                    if CANVAS_READY
                    else gr.HTML(
                        gradio_placeholder(), label="3D constraint scene editor"
                    )
                )
                frame_slider = gr.Slider(
                    0,
                    starter["n_frames"] - 1,
                    step=1,
                    value=int(starter["frame"]),
                    label="current frame",
                    info="editing frame — the armature preview, the underlay and "
                    "the constraint handles all sit here",
                )
            with gr.Column(scale=5):
                prompt_box = gr.Textbox(
                    lines=2,
                    value=starter["prompt"],
                    label="prompt",
                    placeholder="a person drags a chair forward",
                )
                n_frames_slider = gr.Slider(
                    16,
                    1024,
                    step=1,
                    value=starter["n_frames"],
                    label="n_frames (20 fps)",
                )
                duration_box = gr.Number(
                    value=round(starter["n_frames"] / FRAMERATE, 2),
                    precision=2,
                    label="duration (seconds)",
                    interactive=True,
                )
                ref_id_dd = gr.Dropdown(
                    choices=_ref_options(),
                    value=None,
                    label="reference motion (HumanML3D id)",
                    interactive=True,
                    info="pick a motion id to drag from a real pose",
                )
                status = gr.Textbox(
                    value=_common_status(starter),
                    lines=6,
                    label="status",
                    interactive=False,
                )

        with gr.Accordion("Keyframes", open=True):
            kf_dropdown = gr.Dropdown(
                choices=_kf_choices(starter),
                value=_kf_label(starter, starter["sel_kf"]),
                label="keyframes",
                interactive=True,
            )
            with gr.Row():
                add_btn = gr.Button("➕ add keyframe @ current frame", size="sm")
                dup_btn = gr.Button("⧉ duplicate", size="sm")
                del_btn = gr.Button("🗑 delete", size="sm")
            with gr.Row():
                wp_btn = gr.Button(
                    "🧭 waypoint @ current frame", size="sm"
                )
            with gr.Row():
                kf_frame_slider = gr.Slider(
                    0,
                    starter["n_frames"] - 1,
                    step=1,
                    value=int(starter["frame"]),
                    label="keyframe frame (retime)",
                    info="retimes the selected keyframe; the scene follows",
                )
                kf_stage = gr.Radio(list(STAGES), value="middle", label="stage")
            with gr.Row():
                joints_cbg = gr.CheckboxGroup(
                    choices=[joint_label(j) for j in range(NUM_JOINTS)],
                    label="controlled joints",
                    interactive=True,
                )
                axes_cbg = gr.CheckboxGroup(
                    choices=["x", "y", "z"],
                    value=["x", "z"],
                    label="constrained axes",
                    interactive=True,
                )
            with gr.Row():
                x_val = gr.Number(value=None, label="X (m)", precision=4)
                z_val = gr.Number(value=None, label="Z (m)", precision=4)
                y_val = gr.Slider(
                    -0.5, 2.5, step=0.01, value=None, label="Y height (m)"
                )

        with gr.Accordion("Run", open=False):
            with gr.Row():
                checkpoint_dd = gr.Dropdown(
                    choices=names,
                    value=default_name,
                    label="run (checkpoints/<t2m>/<name>)",
                    interactive=True,
                )
                ckpt_dd = gr.Dropdown(
                    choices=ckpts,
                    value=(ckpts[0] if ckpts else None),
                    label="checkpoint file (.tar)",
                    interactive=True,
                )
                device_dd = gr.Dropdown(
                    choices=devices,
                    value=(devices[0] if devices else "cpu"),
                    label="device",
                    interactive=True,
                )
            with gr.Row():
                num_samples = gr.Slider(1, 8, step=1, value=1, label="num_samples")
                num_steps = gr.Slider(2, 1000, step=10, value=300, label="num_steps")
                cfg_box = gr.Number(value=2.0, label="cfg (CFG doubles the batch)")
                w_kin_box = gr.Number(value=1.0, label="w_kin")
                seed_box = gr.Number(value=3407, precision=0, label="seed")
            with gr.Row():
                use_metric_cb = gr.Checkbox(True, label="use_metric_R")
                use_augmented_cb = gr.Checkbox(True, label="use_augmented_obs")
                use_noise_cb = gr.Checkbox(True, label="use_noise_mixing")
                noise_mix_box = gr.Number(value=1.0, label="noise_mix_strength")
                save_mp4_cb = gr.Checkbox(True, label="save .mp4")
                save_gif_cb = gr.Checkbox(False, label="save .gif")
            with gr.Row():
                blender_host_box = gr.Textbox(
                    value="auto",
                    label="Blender host",
                    interactive=True,
                    info="auto = loopback, then the Windows host seen from WSL2; "
                    "needs the bridge receiver running in Blender (N panel)",
                )
                blender_port = gr.Number(
                    value=BLENDER_PORT, precision=0, label="Blender port"
                )
                send_btn = gr.Button(
                    "📡 send to Blender (new action/clip)", size="sm"
                )

        with gr.Row():
            spec_reload = gr.Dropdown(
                choices=_spec_options(),
                value=None,
                label="load spec (bundled / user)",
                interactive=True,
                info="editable copies live under specs/",
            )
            spec_name_box = gr.Textbox(
                value="drag_chair", label="spec name", interactive=True
            )
            save_btn = gr.Button("💾 save spec", size="sm")
            run_btn = gr.Button("▶ run generation", variant="primary")

        spec_state = gr.State(starter)
        out_dir_md = gr.Markdown(visible=False)
        zip_file = gr.File(label="bundle (zip)", visible=False)
        video_slots = [
            gr.Video(label=f"sample {i:02d}", visible=False)
            for i in range(MAX_VIDEO_SLOTS)
        ]
        outputs_by_name = {
            "canvas": canvas_el,
            "kf_dropdown": kf_dropdown,
            "joints_cbg": joints_cbg,
            "axes_cbg": axes_cbg,
            "kf_frame_slider": kf_frame_slider,
            "kf_stage": kf_stage,
            "x_val": x_val,
            "y_val": y_val,
            "z_val": z_val,
            "prompt_box": prompt_box,
            "n_frames_slider": n_frames_slider,
            "duration_box": duration_box,
            "ref_id_dd": ref_id_dd,
            "frame_slider": frame_slider,
            "status": status,
        }
        editor_outputs = [spec_state, *(outputs_by_name[name] for name in EDIT_OUTPUTS)]
        quiet = {"show_progress": "hidden"}  # editor micro-events: no spinner flash
        prompt_box.change(
            on_prompt_changed, [spec_state, prompt_box], editor_outputs, **quiet
        )
        n_frames_slider.change(
            on_n_frames_changed,
            [spec_state, n_frames_slider],
            editor_outputs,
            **quiet,
        )
        duration_box.change(
            on_duration_changed, [spec_state, duration_box], editor_outputs, **quiet
        )
        ref_id_dd.change(
            on_ref_changed, [spec_state, ref_id_dd], editor_outputs, **quiet
        )
        frame_slider.change(
            on_frame_changed, [spec_state, frame_slider], editor_outputs, **quiet
        )
        add_btn.click(
            on_add_keyframe, [spec_state, frame_slider], editor_outputs, **quiet
        )
        wp_btn.click(
            on_add_waypoint, [spec_state, frame_slider], editor_outputs, **quiet
        )
        # every keyframe-scoped handler receives the dropdown's live value so a
        # select→act race can't retarget the edit to a stale keyframe
        dup_btn.click(
            on_duplicate_keyframe, [spec_state, kf_dropdown], editor_outputs, **quiet
        )
        del_btn.click(
            on_delete_keyframe, [spec_state, kf_dropdown], editor_outputs, **quiet
        )
        joints_cbg.change(
            on_joints_changed,
            [spec_state, joints_cbg, kf_dropdown],
            editor_outputs,
            **quiet,
        )
        axes_cbg.change(
            on_axes_changed,
            [spec_state, axes_cbg, kf_dropdown],
            editor_outputs,
            **quiet,
        )
        kf_frame_slider.release(
            on_kf_frame_changed,
            [spec_state, kf_frame_slider, kf_dropdown],
            editor_outputs,
            **quiet,
        )
        kf_stage.change(
            on_stage_changed,
            [spec_state, kf_stage, kf_dropdown],
            editor_outputs,
            **quiet,
        )
        x_val.change(
            on_x_changed, [spec_state, x_val, kf_dropdown], editor_outputs, **quiet
        )
        y_val.change(
            on_y_changed, [spec_state, y_val, kf_dropdown], editor_outputs, **quiet
        )
        z_val.change(
            on_z_changed, [spec_state, z_val, kf_dropdown], editor_outputs, **quiet
        )
        if CANVAS_READY:
            canvas_el.change(
                on_canvas_change,
                [canvas_el, spec_state, kf_dropdown],
                editor_outputs,
                **quiet,
            )
        spec_reload.change(
            on_load_spec, [spec_state, spec_reload], editor_outputs, **quiet
        )
        save_btn.click(
            on_save_spec, [spec_state, spec_name_box], editor_outputs, **quiet
        )
        checkpoint_dd.change(
            on_checkpoint_change, [spec_state, checkpoint_dd], [spec_state, ckpt_dd]
        )

        run_btn.click(
            on_run,
            inputs=[
                spec_state,
                checkpoint_dd,
                ckpt_dd,
                device_dd,
                spec_name_box,
                num_samples,
                num_steps,
                cfg_box,
                w_kin_box,
                seed_box,
                use_metric_cb,
                use_augmented_cb,
                use_noise_cb,
                noise_mix_box,
                save_mp4_cb,
                save_gif_cb,
            ],
            outputs=[spec_state, status, out_dir_md, zip_file, canvas_el, *video_slots],
            queue=True,
        )
        send_btn.click(
            on_send_to_blender,
            inputs=[out_dir_md, blender_host_box, blender_port],
            outputs=[status],
            queue=False,
        )
    return demo


def main() -> None:
    _log(
        f"server build {WEBUI_BUILD} starting — hard-refresh the browser "
        "(Ctrl+Shift+R) so the component bundle matches"
    )
    demo = build_ui()
    demo.queue(max_size=8)
    demo.launch(
        server_name="0.0.0.0",
        server_port=None,
        show_error=True,
        inbrowser=False,
        share=False,
    )


if __name__ == "__main__":
    main()
