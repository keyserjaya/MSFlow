"""Blender headless self-test for the MSFlow motion bridge.

    blender --background --python blender_addon/tests/retarget_selftest.py

Builds a synthetic 22-joint motion (standing ref pose + rigid limb edits +
crouch + whole-body yaw), runs it through the REAL webui exporter, then
imports/retargets it onto a straight-bone test rig built from the same ref
pose — so proportions match and retargeted bone heads must land on the
source joints. Also exercises name auto-match, duration mapping, fractional
keyframe timing, and the flip-facing/in-place variants.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import bpy  # noqa: E402
import numpy as np  # noqa: E402 (Blender bundles numpy)

from blender_addon.msflow_bridge import import_ops, io_msflow, mapping, retarget  # noqa: E402
from webui.msflow_export import build_motion_dict, joint_parents  # noqa: E402
from webui.skeleton import canonical_pose  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    if condition:
        print(f"  ok: {name}")
    else:
        FAILURES.append(f"{name} ({detail})")
        print(f"  FAIL: {name} ({detail})")
    return bool(condition)


def close(value: float, expected: float, tol: float) -> bool:
    return abs(float(value) - float(expected)) <= tol


def synthetic_positions(t_total: int = 41) -> np.ndarray:
    """[T, 22, 3] standing pose with rigid edits, in MSFlow world space."""
    base = canonical_pose().astype(np.float32)
    frames = np.repeat(base[None, :, :], t_total, axis=0).copy()
    identity = np.eye(3, dtype=np.float32)

    def rotation(axis: str, amount: float) -> np.ndarray:
        matrix = identity.copy()
        c, s = math.cos(amount), math.sin(amount)
        if axis == "Z":
            matrix[:2, :2] = [[c, -s], [s, c]]
        elif axis == "X":
            matrix[1:, 1:] = [[c, -s], [s, c]]
        return matrix

    def rotate_subtree(pose: np.ndarray, members, pivot: np.ndarray, rot) -> None:
        for member in members:
            pose[member] = rot @ (pose[member] - pivot) + pivot

    for index in range(t_total):
        alpha = index / (t_total - 1)
        pose = frames[index]
        yaw = rotation("Z", math.radians(30.0) * alpha)
        rotate_subtree(pose, tuple(range(1, 22)), base[0], yaw)
        lift = rotation("X", math.radians(60.0) * math.sin(math.pi * alpha))
        rotate_subtree(pose, (18, 20), base[16], lift)
        swing = rotation("X", math.radians(20.0) * math.sin(math.pi * alpha))
        rotate_subtree(pose, (8, 11), base[5], swing)
        pose[:, 1] += -0.15 * math.sin(math.pi * alpha)
    return frames


# bone name, (head joint, tail joint), pose parent; (15,15)/(20,20)/(21,21)
# are rest-leaf bones whose tails extend along a proximal segment.
RIG_BONES: dict[int, tuple[str, tuple[int, int], str | None]] = {
    0: ("hips", (0, 3), None),
    3: ("spine", (3, 6), "hips"),
    6: ("chest", (6, 9), "spine"),
    9: ("upper_chest", (9, 12), "chest"),
    12: ("neck", (12, 15), "upper_chest"),
    15: ("head", (15, 15), "neck"),
    13: ("clavicle.L", (13, 16), "upper_chest"),
    16: ("upper_arm.L", (16, 18), "clavicle.L"),
    18: ("forearm.L", (18, 20), "upper_arm.L"),
    20: ("hand.L", (20, 20), "forearm.L"),
    14: ("clavicle.R", (14, 17), "upper_chest"),
    17: ("upper_arm.R", (17, 19), "clavicle.R"),
    19: ("forearm.R", (19, 21), "upper_arm.R"),
    21: ("hand.R", (21, 21), "forearm.R"),
    1: ("thigh.L", (1, 4), "hips"),
    4: ("shin.L", (4, 7), "thigh.L"),
    7: ("foot.L", (7, 10), "shin.L"),
    10: ("toe.L", (10, 10), "foot.L"),
    2: ("thigh.R", (2, 5), "hips"),
    5: ("shin.R", (5, 8), "thigh.R"),
    8: ("foot.R", (8, 11), "shin.R"),
    11: ("toe.R", (11, 11), "foot.R"),
}

EXPECTED_MATCH = {
    0: "hips", 3: "spine", 6: "chest", 9: "upper_chest", 12: "neck",
    13: "clavicle.L", 14: "clavicle.R", 16: "upper_arm.L", 17: "upper_arm.R",
    18: "forearm.L", 19: "forearm.R", 20: "hand.L", 21: "hand.R",
    1: "thigh.L", 2: "thigh.R", 4: "shin.L", 5: "shin.R", 7: "foot.L",
    8: "foot.R", 10: "toe.L", 11: "toe.R", 15: "head",
}


def build_test_armature(reference_bl: list) -> bpy.types.Object:
    """Straight-bone rig (Rigify-ish names) posed like the reference frame."""
    armature = bpy.data.armatures.new("MSFlowTestRig")
    obj = bpy.data.objects.new("MSFlowTestRig", armature)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    created: dict[str, bpy.types.EditBone] = {}

    def leaf_extension(joint: int):
        if joint == 15:
            return reference_bl[15] - reference_bl[12]
        if joint == 20:
            return reference_bl[20] - reference_bl[18]
        if joint == 21:
            return reference_bl[21] - reference_bl[19]
        proximal = 7 if joint == 10 else 8
        return reference_bl[joint] - reference_bl[proximal]  # toes on the foot

    for joint in sorted(RIG_BONES):
        name, (head, tail), parent_name = RIG_BONES[joint]
        bone = armature.edit_bones.new(name)
        bone.head = tuple(reference_bl[head])
        if head == tail:
            direction = leaf_extension(joint).normalized() * 0.12
            bone.tail = tuple(reference_bl[head] + direction)
        else:
            bone.tail = tuple(reference_bl[tail])
        if parent_name is not None:
            bone.parent = created[parent_name]
            bone.use_connect = (created[parent_name].tail - bone.head).length < 1e-5
        created[name] = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def current_head(obj: bpy.types.Object, bone: str):
    for pose_bone in obj.pose.bones:
        if pose_bone.name == bone:
            return pose_bone.matrix.to_translation()
    raise KeyError(bone)


def main() -> int:
    print("== MSFlow bridge selftest ==")

    converted = io_msflow.to_blender_space(np.array([[0.3, 1.2, -0.5]], dtype="float32"))
    check("axis Y-up -> Z-up", np.allclose(converted[0], [0.3, 0.5, 1.2]), str(converted[0]))
    flipped = io_msflow.to_blender_space(np.array([[0.3, 1.2, -0.5]], dtype="float32"), True)
    check("flip facing", np.allclose(flipped[0], [-0.3, -0.5, 1.2]), str(flipped[0]))

    try:
        io_msflow.parse_motion_dict({"format": "nope", "version": 1})
        check("format guard", False, "accepted bad format")
    except io_msflow.MsflowFormatError:
        check("format guard", True)

    decoys = ["MCH-thigh.L", "WGT-upper_arm.L", "thigh_ik.L", "thigh_twist.L"]
    bones = [*decoys, *(entry[0] for entry in RIG_BONES.values())]
    match = mapping.auto_match(bones)
    check(
        "auto-match",
        all(match.get(joint) == name for joint, name in EXPECTED_MATCH.items()),
        json.dumps({str(joint): match.get(joint) for joint in EXPECTED_MATCH}),
    )
    check("auto-match skips mechanical", "MCH-thigh.L" not in match.values())

    successors = retarget.chain_successors(joint_parents())
    check("chain succ pelvis", successors.get(0) == 3, str(successors))
    check("chain succ neck", successors.get(12) == 15, str(successors))
    check("chain succ hip", successors.get(1) == 4, str(successors))
    check("chain succ wrist leaf", 20 not in successors and 15 not in successors)

    positions = synthetic_positions()
    controls = [
        {
            "joints": [21, 0],
            "frame": 5,
            "axes": "x,y,z",
            "stage": "early",
            "values": [[0.2, 0.9, 0.1], [0.0, 0.95, 0.0]],
        },
        {
            "joints": [20],
            "frame": 25,
            "axes": "x,z",
            "stage": "late",
            "role": "waypoint",
            "values": [[-0.2, 0.9, 0.1]],
        },
    ]
    with tempfile.TemporaryDirectory() as tmp_name:
        run_dir = Path(tmp_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "sample_00.msflow.json").write_text(
            json.dumps(build_motion_dict(positions, controls=controls)),
            encoding="utf-8",
        )
        motion = io_msflow.load_motion(run_dir / "sample_00.msflow.json")
        check("loader roundtrip", motion.n_frames == 41 and motion.fps == 20.0)
        frames_bl = retarget.convert_positions(motion, False)
        obj = build_test_armature(frames_bl[0])

        import_ops.register_scene_props()
        config = bpy.context.scene.msflow_bridge_settings
        config.target = obj
        for joint, name in EXPECTED_MATCH.items():
            setattr(config, mapping.joint_field_name(joint), name)

        scene = bpy.context.scene
        scene.render.fps = 24
        options = retarget.RetargetOptions(root_motion="hips")
        report = retarget.retarget_motion(motion, obj, EXPECTED_MATCH, options, scene)
        check("action named", report.action_name.startswith("MSFlow_"), report.action_name)
        markers = bpy.data.collections.get(retarget.CONSTRAINT_COLLECTION)
        check(
            "constraint empties imported",
            markers is not None and len(markers.objects) == 3,
            str(None if markers is None else len(markers.objects)),
        )
        names = [item.name for item in markers.objects] if markers else []
        check("waypoint markers named", any(name.startswith("MSF_wp_") for name in names), str(names))
        check(
            "leg scale 1 on identical rig",
            close(report.scale, 1.0, 1e-6),
            str(report.scale),
        )
        check("scene fps untouched", scene.render.fps == 24, str(scene.render.fps))
        check("frame range fits duration", scene.frame_end == 49, str(scene.frame_end))
        check(
            "fk deviation small",
            report.mean_error_cm < 0.5,
            f"mean {report.mean_error_cm:.3f}cm max {report.max_error_cm:.3f}cm",
        )

        fcurves = obj.animation_data.action.fcurves
        thigh = fcurves.find('pose.bones["thigh.L"].rotation_quaternion', 0)
        if check("quaternion keys exist", thigh is not None) and thigh is not None:
            active = [point.co[0] for point in thigh.keyframe_points]
            expected_frames = [1.0 + index * (24.0 / 20.0) for index in range(motion.n_frames)]
            timing_ok = len(active) == motion.n_frames and all(
                close(frame, expected, 1e-4)
                for frame, expected in zip(active, expected_frames)
            )
            check("fractional key timing", timing_ok, json.dumps(active[:4]))

        follow = True
        rest_ok = True
        for index in (0, 20, 40):
            key = 1.0 + index * 1.2
            scene.frame_set(int(key), subframe=key - int(key))
            bpy.context.view_layer.update()
            for joint, (name, _span, _parent) in RIG_BONES.items():
                got = current_head(obj, name)
                want = frames_bl[index][joint]
                if (got - want).length > 0.02:
                    follow = False
                    print(
                        f"    frame {index} joint {joint} {name}: "
                        f"{got[:]} vs {want[:]}"
                    )
                if index == 0:
                    rest = obj.pose.bones[name].bone.matrix_local.to_translation()
                    if (got - rest).length > 1e-4:
                        rest_ok = False
                        print(f"    frame 1 {name}: pose {got[:]} != rest {rest[:]}")
        check("fk heads follow source", follow)
        check("frame 1 == rest pose", rest_ok)

        options = retarget.RetargetOptions(root_motion="in_place", flip_facing=True)
        variant = retarget.retarget_motion(motion, obj, EXPECTED_MATCH, options, scene)
        check("variant run completes", variant.action_name != report.action_name)

    if FAILURES:
        print(f"selftest FAILED: {len(FAILURES)}")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print("selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
