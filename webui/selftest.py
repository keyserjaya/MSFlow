"""Headless self-test for the webui (no GPU, no server needed).

Run from the repo root:
    uv run --group ui python webui/selftest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webui import app as app_module  # noqa: E402
from webui.skeleton import JOINT_CHAIN, NUM_JOINTS  # noqa: E402
from webui.spec_schema import load_spec_file  # noqa: E402


def canvas_slot_is_value(out: tuple) -> bool:
    """A refresh tuple's canvas slot returns a real CanvasValue (has
    'underlay') only for server-driven scene changes; gr.update() no-ops are
    dicts with a '__type__' sentinel in this Gradio version."""
    slot = out[1]
    return isinstance(slot, dict) and "underlay" in slot


def check_skeleton() -> None:
    chain_joints = {j for pair in JOINT_CHAIN for j in pair}
    assert all(0 <= j < NUM_JOINTS for j in chain_joints), "chain has bad joint ids"
    assert len(app_module.JOINT_NAMES) == NUM_JOINTS
    print("skeleton ok (chain uses", len(chain_joints), "joints)")


def check_roundtrip() -> None:
    bundled = Path(__file__).resolve().parent.parent / "sample" / "joint_control_specs"
    for path in sorted(bundled.glob("*.json")):
        spec = load_spec_file(path)
        import tempfile

        tmp = Path(tempfile.mkstemp(suffix=".json")[1])
        spec.dump(tmp)
        reparsed = load_spec_file(tmp)
        assert json.loads(tmp.read_text()) == json.loads(
            path.read_text()
        ), f"round-trip mismatch for {path.name}"
        assert [c.frame for c in reparsed.controls] == [c.frame for c in spec.controls]
        tmp.unlink(missing_ok=True)
        print(f"round-trip ok: {path.name} ({len(spec.controls)} controls)")


def check_build() -> None:
    demo = app_module.build_ui()
    assert demo is not None
    print(f"build_ui ok (canvas_ready={app_module.CANVAS_READY})")


def check_editor_contract() -> None:
    state = app_module.new_state()
    out = app_module.refresh(state)
    assert len(out) == 1 + len(app_module.EDIT_OUTPUTS), len(out)
    out = app_module.refresh(state, light=True)
    assert len(out) == 1 + len(app_module.EDIT_OUTPUTS), len(out)
    app_module.on_duration_changed(state, 3.0)
    assert state["n_frames"] == 60, state["n_frames"]
    out = app_module.refresh(state, light=True, kf_frame_bounds=True)
    assert len(out) == 1 + len(app_module.EDIT_OUTPUTS), len(out)
    state["playback"] = {"frames": [[[0.0] * 3] * 22], "fps": 20, "label": "t"}
    value = app_module.canvas_value(state)
    assert value["playback"] and len(value["playback"]["frames"]) == 1
    assert value["frame"] == state["frame"]
    print("editor contract ok (arity, duration sync, playback passthrough)")


def check_canvas_value() -> None:
    state = app_module.new_state()
    value = app_module.canvas_value(state)
    assert value["frame"] == 0, value["frame"]
    ids = {m["id"] for m in value["markers"]}
    expected = {f"f{f}-j{j}" for f in (0, 60, 120) for j in (20, 21)}
    assert ids == expected, ids  # every frame's constraints, not just the selected one
    for marker in value["markers"]:
        assert sorted(marker["axes"]) == ["x", "z"], marker  # starter axes
        assert marker["selected"] == (
            marker["frame"] == 0 and marker["joint"] == state["sel_joint"]
        )
    assert len(value["underlay"]["points"]) == NUM_JOINTS
    print("canvas value ok (all-frame markers with id/frame/axes, frame field)")


def check_canvas_handler() -> None:
    state = app_module.new_state()
    control = state["controls"][state["sel_kf"]]
    row_index = control["joints"].index(21)
    out = app_module.on_canvas_change(
        {
            "action": "move",
            "joint": 21,
            "x": 0.5,
            "y": 1.2,
            "z": -0.3,
            "axes": ["x", "y"],
        },
        state,
    )
    row = control["values"][row_index]
    assert abs(row[0] - 0.5) < 1e-6, row
    assert abs(row[1] - 1.2) < 1e-6, row
    # the FULL dropped position is stored even for untouched axes — the drag's
    # solved pose must not snap back to the reference seed on the next sync
    assert abs(row[2] - (-0.3)) < 1e-6, row
    assert 0 in control["axes"] and 1 in control["axes"], control["axes"]
    assert not canvas_slot_is_value(out), "move must not re-emit the canvas value"

    # frame resolution: with the selection frame mismatched, the edit must land
    # in a keyframe at the *current* frame, never in a stale one
    state["frame"] = 50
    app_module.on_canvas_change(
        {
            "action": "move",
            "joint": 3,
            "x": 0.1,
            "y": 1.0,
            "z": 0.2,
            "axes": ["x", "y", "z"],
        },
        state,
    )
    assert 3 in state["controls"][state["sel_kf"]]["joints"]
    assert state["controls"][state["sel_kf"]]["frame"] == 50, state["controls"][
        state["sel_kf"]
    ]
    state["frame"], state["sel_kf"] = 0, 0

    out = app_module.on_canvas_change({"action": "select", "joint": -1}, state)
    assert state["sel_joint"] is None
    assert not canvas_slot_is_value(out), "select must not re-emit the canvas value"

    out = app_module.on_canvas_change(
        {"action": "select", "joint": 20, "frame": 60}, state
    )
    assert state["frame"] == 60, state["frame"]  # ghost jump switches the frame
    assert state["controls"][state["sel_kf"]]["frame"] == 60
    assert 20 in state["controls"][state["sel_kf"]]["joints"]
    assert canvas_slot_is_value(out) and out[1]["frame"] == 60, out[1]

    state = app_module.new_state()  # fresh state for the snapshot
    pose = [[0.0, 0.9, 0.0] for _ in range(NUM_JOINTS)]
    pose[21] = [0.3, 0.8, -0.2]
    # with a keyframe selected, the snapshot replaces THAT keyframe at its own
    # frame (a past frame stays a past frame) even if the payload says otherwise
    label = app_module._kf_label(state, 1)
    out = app_module.on_canvas_change(
        {"action": "snapshot", "frame": 77, "pose": pose}, state, label
    )
    at_60 = [i for i, c in enumerate(state["controls"]) if c["frame"] == 60]
    assert len(at_60) == 1, at_60  # replaced every constraint at that frame
    assert not [c for c in state["controls"] if c["frame"] == 77]  # no new frame
    pinned = state["controls"][at_60[0]]
    assert pinned["joints"] == list(range(NUM_JOINTS))
    assert pinned["values"][21] == [0.3, 0.8, -0.2]
    assert pinned["axes"] == [0, 1, 2]
    assert state["sel_kf"] == at_60[0]
    assert canvas_slot_is_value(out), "snapshot must re-emit the canvas value"

    # with NO keyframe selected, the snapshot works at the payload's frame
    state = app_module.new_state()
    state["sel_kf"] = -1
    app_module.on_canvas_change(
        {"action": "snapshot", "frame": 77, "pose": pose}, state
    )
    assert len([c for c in state["controls"] if c["frame"] == 77]) == 1
    print("canvas handler ok (axis-aware move, frame resolution, ghost jump, snapshot)")


def check_keyframe_selection() -> None:
    """The dropdown's live value must win over a stale state sel_kf, and
    scrubbing the current frame must not steal the keyframe selection."""
    # select→act race: the move handler leaves sel_kf at the LAST keyframe;
    # acting on item[1] via its dropdown label must still hit item[1]
    state = app_module.new_state()
    state["sel_kf"] = len(state["controls"]) - 1
    label = app_module._kf_label(state, 1)
    app_module.on_delete_keyframe(state, label)
    frames = [c["frame"] for c in state["controls"]]
    assert frames == [0, 120], frames  # item[1] (frame 60) deleted, not the last

    state = app_module.new_state()
    state["sel_kf"] = len(state["controls"]) - 1
    label = app_module._kf_label(state, 1)
    app_module.on_duplicate_keyframe(state, label)
    assert state["sel_kf"] == 2 and state["controls"][2]["frame"] == 60

    state = app_module.new_state()
    state["sel_kf"] = len(state["controls"]) - 1
    label = app_module._kf_label(state, 1)
    app_module.on_kf_frame_changed(state, 33, label)
    assert state["controls"][1]["frame"] == 33, state["controls"][1]
    assert state["controls"][2]["frame"] == 120  # the last item untouched

    # scrubbing the current frame keeps the keyframe selection
    state = app_module.new_state()
    state["sel_kf"] = 1
    app_module.on_frame_changed(state, 77)
    assert state["sel_kf"] == 1, state["sel_kf"]
    print("keyframe selection ok (label wins over stale state; scrub keeps it)")


def check_echo_shortcircuits() -> None:
    """Programmatic panel echoes must never re-emit the canvas value (each one
    used to rebuild the scene and reset freshly dragged constraints)."""
    state = app_module.new_state()
    label = app_module._kf_label(state, state["sel_kf"])
    out = app_module.on_kf_select(state, label)
    assert not canvas_slot_is_value(out), "kf echo must not re-emit canvas"
    out = app_module.on_frame_changed(state, state["frame"])
    assert not canvas_slot_is_value(out), "frame echo must not re-emit canvas"
    control = state["controls"][state["sel_kf"]]
    joints_labels = [app_module.joint_label(j) for j in control["joints"]]
    out = app_module.on_joints_changed(state, joints_labels)
    assert not canvas_slot_is_value(out), "joints echo must not re-emit canvas"
    axes_labels = [app_module.AXES_BY_INDEX[a] for a in control["axes"]]
    out = app_module.on_axes_changed(state, axes_labels)
    assert not canvas_slot_is_value(out), "axes echo must not re-emit canvas"
    out = app_module.on_stage_changed(state, control.get("stage") or "middle")
    assert not canvas_slot_is_value(out), "stage echo must not re-emit canvas"
    out = app_module.on_duration_changed(
        state, state["n_frames"] / app_module.FRAMERATE
    )
    assert not canvas_slot_is_value(out), "duration echo must not re-emit canvas"

    # widget-precision echoes must short-circuit (float32 rows vs display
    # rounding: the y slider returns 0.9 for a stored 0.8999999761581421)
    state = app_module.new_state()
    control = state["controls"][state["sel_kf"]]
    state["sel_joint"] = control["joints"][0]
    row = control["values"][0]
    out = app_module.on_value_changed(state, 1, round(row[1], 2))
    assert not canvas_slot_is_value(out), "y echo at widget precision must not rebuild"

    # a value-wrapped dispatch payload must still reach the handler
    state = app_module.new_state()
    app_module.on_canvas_change(
        {"value": {"action": "select", "joint": 7, "frame": 0}}, state
    )
    assert state["sel_joint"] == 7, state["sel_joint"]

    # the move payload's frame is authoritative: the fold must land there
    state = app_module.new_state()
    state["frame"] = 10
    app_module.on_canvas_change(
        {
            "action": "move",
            "joint": 21,
            "x": 1.0,
            "y": 1.0,
            "z": 1.0,
            "axes": ["x"],
            "frame": 30,
        },
        state,
    )
    assert state["frame"] == 30, state["frame"]
    control = app_module._selected_control(state)
    assert control is not None and control["frame"] == 30, control
    print(
        "echo short-circuits ok (panel echoes never rebuild the scene); "
        "move frame is client-authoritative"
    )


def check_reference() -> None:
    ids = app_module.reference.list_ids()
    print(
        f"reference ok: present={app_module.reference.available} ids_found={len(ids)}"
    )


if __name__ == "__main__":
    check_skeleton()
    check_roundtrip()
    check_reference()
    check_canvas_value()
    check_canvas_handler()
    check_keyframe_selection()
    check_echo_shortcircuits()
    check_editor_contract()
    check_build()
    print("selftest OK")
