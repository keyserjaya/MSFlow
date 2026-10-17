"""Retarget `msflow-motion` data onto a Blender armature (bpy + mathutils).

Per mapped target bone, each source frame provides:
- +Y direction: anchor joint -> its chain successor (the anatomical
  continuation: pelvis->spine, hip->knee->ankle->toe, neck->head,
  collar->shoulder->elbow->wrist); leaf bones (hands/head) inherit the
  nearest mapped ancestor's direction so they ride along rigidly;
- roll: the continuing chain's next segment, else the mapped parent's roll
  carrier, else a world axis — straight chains and leaves stay stable and
  never candy-wrapper;
- the hips/root frame is built directly from the pelvis trunk (`0->3`) plus
  the hip line (`1->2`), which is the only cue that carries torso yaw/twist;
- world rotation delta = F(t) @ F(ref)^-1, composed with the target's rest
  orientation, keeping the rig's rest pose at the reference frame;
- head positions from target-proportional FK (parents are set first); the
  mapped hips/root keyframes the (leg-scale-matched) pelvis translation.

Keys land at 1 + i * scene_fps / source_fps (fractional frames allowed, so
timing stays in real seconds at any scene frame rate).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import bpy
from mathutils import Matrix, Vector

from .io_msflow import (
    MotionData,
    default_action_label,
    to_blender_space,
)
from .mapping import missing_required

EPS = 1e-8
CONSTRAINT_COLLECTION = "MSFlow Constraints"


@dataclass
class RetargetOptions:
    root_motion: str = "hips"  # "hips" | "object" | "in_place"
    scale_match: bool = True
    flip_facing: bool = False
    integer_resample: bool = False
    set_frame_range: bool = True
    nla_clip: bool = False
    constraint_empties: bool = True
    ref_frame: int = 0


@dataclass
class RetargetReport:
    action_name: str
    frames: int
    duration_sec: float
    mapped: dict[int, str]
    missing: list[int]
    scale: float
    mean_error_cm: float
    max_error_cm: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class BoneBinding:
    joint: int  # head anchor joint (the mapping key)
    bone_name: str
    pose_bone: bpy.types.PoseBone
    rest_arm: Matrix  # armature-space rest matrix
    tail_joint: int | None  # +Y tracks anchor->tail_joint (None = leaf)
    parent_binding: "BoneBinding | None" = None
    offset_local: Vector = field(default_factory=lambda: Vector((0.0, 0.0, 0.0)))

    @property
    def rest_rot(self) -> Matrix:
        return self.rest_arm.to_3x3()

    @property
    def rest_head(self) -> Vector:
        return self.rest_arm.to_translation()


# Chain continuations that beat the exported tree's parenthood: the pelvis
# continues into the spine (not the leg), the neck continues to the head
# (not the collars).
SECTION_OVERRIDES = {0: 3, 12: 15}


def chain_successors(parents: list[int]) -> dict[int, int]:
    """Child joint of each joint along its anatomical continuation chain."""
    children: dict[int, list[int]] = {joint: [] for joint in range(len(parents))}
    for joint, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(joint)
    successors: dict[int, int] = {}
    for joint, kids in children.items():
        if joint in SECTION_OVERRIDES:
            successors[joint] = SECTION_OVERRIDES[joint]
        elif kids:
            successors[joint] = min(kids)
    return successors


def build_bindings(
    armature_obj: bpy.types.Object, mapping: dict[int, str], parents: list[int]
) -> list[BoneBinding]:
    """Bindings in FK order (parents before children); every mapped pose
    bone switches to quaternion rotation and prepares its FK offset."""
    pose_bones = armature_obj.pose.bones
    successors = chain_successors(parents)
    by_name: dict[str, BoneBinding] = {}
    bindings: list[BoneBinding] = []
    for joint in sorted(mapping):
        name = mapping[joint]
        if name not in pose_bones:
            raise ValueError(f"target rig has no bone '{name}' (joint {joint})")
        pose_bone = pose_bones[name]
        pose_bone.rotation_mode = "QUATERNION"
        rest_arm = pose_bone.bone.matrix_local.copy()
        parent_binding = None
        walker = pose_bone.parent
        while walker is not None and parent_binding is None:
            parent_binding = by_name.get(walker.name)
            walker = walker.parent
        binding = BoneBinding(
            joint=joint,
            bone_name=name,
            pose_bone=pose_bone,
            rest_arm=rest_arm,
            tail_joint=successors.get(joint),
            parent_binding=parent_binding,
        )
        if parent_binding is not None:
            parent_rest = parent_binding.rest_arm
            binding.offset_local = parent_rest.to_3x3().inverted_safe() @ (
                binding.rest_head - parent_rest.to_translation()
            )
        by_name[name] = binding
        bindings.append(binding)

    def depth(binding: BoneBinding) -> int:
        walker = binding.pose_bone.parent
        count = 0
        while walker is not None:
            count += 1
            walker = walker.parent
        return count

    return sorted(bindings, key=depth)


def segment_frame(
    direction: Vector,
    refs: list[Vector],
    label: str,
    frame: int,
) -> tuple[Matrix | None, Vector | None]:
    """Right-handed rotation with local +Y along `direction`; `refs` are
    ordered roll candidates in world space. Returns (matrix, roll carrier x)
    — the carrier replays this bone's chosen roll into descendant bindings."""
    length = direction.length
    if length < EPS:
        return None, None
    y = direction / length
    for candidate in [*refs, Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0))]:
        if candidate.length < EPS:
            continue
        z = candidate.cross(y)
        if z.length < 1e-6:
            continue
        z.normalize()
        x = y.cross(z)
        x.normalize()
        matrix = Matrix(((x.x, y.x, z.x), (x.y, y.y, z.y), (x.z, y.z, z.z)))
        return matrix, x.copy()
    print(f"[msflow] frame {frame}: no valid roll reference for {label}")
    return None, None


def pelvis_frame(pos: list[Vector]) -> tuple[Matrix | None, Vector | None]:
    """Hips/root orientation from the trunk direction and the hip line, the
    only position-based cue for torso yaw/twist."""
    trunk = pos[3] - pos[0]
    hip_line = pos[2] - pos[1]
    if trunk.length < EPS:
        return None, None
    y = trunk.normalized()
    if hip_line.length < 1e-6 or abs(hip_line.normalized().dot(y)) > 0.999:
        return segment_frame(y, [], "pelvis", 0)
    z = hip_line.cross(y)
    z.normalize()
    x = y.cross(z)
    x.normalize()
    matrix = Matrix(((x.x, y.x, z.x), (x.y, y.y, z.y), (x.z, y.z, z.z)))
    return matrix, x.copy()


def convert_positions(motion: MotionData, flip_facing: bool) -> list[list[Vector]]:
    array = to_blender_space(motion.positions, flip_facing)
    return [[Vector(row) for row in frame] for frame in array]


def leg_scale(
    positions_bl: list[list[Vector]],
    armature_obj: bpy.types.Object,
    mapping: dict[int, str],
) -> float:
    """Uniform target/source ratio from the hip-to-ankle (thigh+shin) span."""
    source = (
        (positions_bl[0][1] - positions_bl[0][7]).length
        + (positions_bl[0][2] - positions_bl[0][8]).length
    ) / 2.0
    target = 0.0
    count = 0
    for joint, ankle in ((1, 7), (2, 8)):
        thigh = mapping.get(joint)
        foot = mapping.get(ankle)
        if not thigh or not foot:
            continue
        if thigh not in armature_obj.pose.bones or foot not in armature_obj.pose.bones:
            continue
        thigh_head = armature_obj.pose.bones[thigh].bone.matrix_local.to_translation()
        ankle_head = armature_obj.pose.bones[foot].bone.matrix_local.to_translation()
        target += (thigh_head - ankle_head).length
        count += 1
    if source < EPS or count == 0 or target / count < EPS:
        return 1.0
    return (target / count) / source


def _ensure_action_slot(animation_data: bpy.types.AnimData) -> None:
    """Slotted actions (Blender 4.4+) need a slot for the object before
    pose-bone keys can land; older versions have no `slots` at all."""
    action = animation_data.action
    if action is None:
        return
    slots = getattr(action, "slots", None)
    if slots is None:
        return
    try:
        if not len(slots):
            animation_data.action_slot = slots.new(id_type="OBJECT")
        elif animation_data.action_slot is None:
            animation_data.action_slot = slots[0]
    except (AttributeError, TypeError):
        return


def retarget_motion(
    motion: MotionData,
    armature_obj: bpy.types.Object,
    mapping: dict[int, str],
    options: RetargetOptions,
    scene: bpy.types.Scene | None = None,
) -> RetargetReport:
    missing = missing_required(mapping)
    if missing:
        raise ValueError(
            "mapping incomplete - missing joints (set bones in the panel): "
            + ", ".join(f"{joint} {motion.joint_names[joint]}" for joint in missing)
        )
    if scene is None:
        scene = bpy.context.scene

    fps_dst = float(scene.render.fps) / (scene.render.fps_base or 1.0) or 24.0
    fps_src = motion.fps
    ref = min(max(int(options.ref_frame), 0), motion.n_frames - 1)
    positions_bl = convert_positions(motion, options.flip_facing)
    scale = leg_scale(positions_bl, armature_obj, mapping) if options.scale_match else 1.0
    bindings = build_bindings(armature_obj, mapping, motion.parents)
    successors = chain_successors(motion.parents)
    hips_binding = next((binding for binding in bindings if binding.joint == 0), None)

    action = bpy.data.actions.new(default_action_label(motion.meta))
    action.use_fake_user = True
    animation_data = armature_obj.animation_data_create()
    animation_data.action = action
    _ensure_action_slot(animation_data)

    track = None
    if options.nla_clip:
        track = animation_data.nla_tracks.new()
        track.name = "MSFlow"

    frames_ref: dict[str, Matrix] = {}
    frames_prev: dict[str, Matrix] = {}
    roll_carriers: dict[str, Vector] = {}
    anchors_ref: dict[str, Vector] = {}
    errors: list[float] = []
    sampled = sorted({ref, motion.n_frames // 2, motion.n_frames - 1})
    object_base = armature_obj.matrix_world.to_translation().copy()

    for index in range(motion.n_frames):
        pos = positions_bl[index]
        key_frame = (
            float(round(1.0 + index * (fps_dst / fps_src)))
            if options.integer_resample
            else 1.0 + index * (fps_dst / fps_src)
        )
        for binding in bindings:
            anchor = pos[binding.joint]
            refs: list[Vector] = []
            if binding.joint == 0:
                # hips/root: trunk + hip line carries torso yaw/twist
                matrix, carrier = pelvis_frame(pos)
            elif binding.tail_joint is not None:
                tail_pos = pos[binding.tail_joint]
                direction = tail_pos - anchor
                beyond = successors.get(binding.tail_joint)
                if beyond is not None:
                    refs.append(pos[beyond] - tail_pos)
                if binding.parent_binding is not None:
                    carrier = roll_carriers.get(binding.parent_binding.bone_name)
                    if carrier is not None:
                        refs.append(carrier)
                matrix, carrier = segment_frame(
                    direction, refs, binding.bone_name, index
                )
            else:
                # leaf bone: ride on its nearest mapped ancestor's frame
                matrix = carrier = None
                if binding.parent_binding is not None:
                    parent_frame = frames_prev.get(binding.parent_binding.bone_name)
                    if parent_frame is not None:
                        matrix = parent_frame.copy()
                        carrier = parent_frame.col[1].copy()

            if matrix is None or carrier is None:
                matrix = frames_prev.get(binding.bone_name)
                if matrix is None:
                    matrix = Matrix.Identity(3)
                    carrier = Vector((1.0, 0.0, 0.0))
            roll_carriers[binding.bone_name] = carrier
            frames_prev[binding.bone_name] = matrix.copy()
            if index == ref:
                frames_ref[binding.bone_name] = matrix.copy()
            delta = matrix @ frames_ref[binding.bone_name].inverted_safe()
            world_rot = delta @ binding.rest_rot

            parent = binding.parent_binding
            if parent is not None:
                parent_pose = parent.pose_bone.matrix
                head = (
                    parent_pose.to_3x3() @ binding.offset_local
                    + parent_pose.to_translation()
                )
            else:
                head = binding.rest_head.copy()

            if binding is hips_binding:
                pelvis_delta = (pos[0] - positions_bl[ref][0]) * scale
                if options.root_motion == "in_place":
                    pelvis_delta = Vector((0.0, 0.0, pelvis_delta.z))
                if options.root_motion == "object":
                    armature_obj.location = object_base + pelvis_delta
                else:
                    head = head + pelvis_delta

            binding.pose_bone.matrix = Matrix.Translation(head) @ world_rot.to_4x4()

        for binding in bindings:
            pose_bone = binding.pose_bone
            pose_bone.keyframe_insert(
                "rotation_quaternion", frame=key_frame, group=binding.bone_name
            )
            if binding is hips_binding and options.root_motion == "hips":
                pose_bone.keyframe_insert(
                    "location", frame=key_frame, group=binding.bone_name
                )
        if options.root_motion == "object" and hips_binding is not None:
            armature_obj.keyframe_insert("location", frame=key_frame)

        if index in sampled:
            for binding in bindings:
                head_world = (
                    armature_obj.matrix_world
                    @ binding.pose_bone.matrix.to_translation()
                )
                if index == ref:
                    anchors_ref[binding.bone_name] = (
                        head_world - pos[binding.joint] * scale
                    )
                    continue
                expected = pos[binding.joint] * scale + anchors_ref[binding.bone_name]
                errors.append((head_world - expected).length)

    if options.nla_clip and track is not None:
        track.strips.new(action.name, 1, action)
        track.mute = True

    if options.set_frame_range:
        last = 1.0 + (motion.n_frames - 1) * (fps_dst / fps_src)
        scene.frame_start = 1
        scene.frame_end = int(math.ceil(last))

    if options.constraint_empties:
        create_constraint_empties(motion, scale, options.flip_facing)

    if len(action.fcurves) == 0:
        animation_data.action = None
        bpy.data.actions.remove(action)
        raise RuntimeError("no fcurves were created (action slot incompatibility?)")

    warnings: list[str] = []
    if motion.duration_sec <= 0.05:
        warnings.append("motion is under 0.05s - check n_frames in the spec")
    mean_cm = 100.0 * sum(errors) / len(errors) if errors else 0.0
    max_cm = 100.0 * max(errors) if errors else 0.0
    return RetargetReport(
        action_name=action.name,
        frames=motion.n_frames,
        duration_sec=motion.duration_sec,
        mapped={joint: mapping[joint] for joint in sorted(mapping)},
        missing=[],
        scale=scale,
        mean_error_cm=mean_cm,
        max_error_cm=max_cm,
        warnings=warnings,
    )


def create_constraint_empties(
    motion: MotionData, scale: float, flip_facing: bool
) -> int:
    """One sphere empty per constrained joint/frame in a dedicated collection."""
    collection = bpy.data.collections.get(CONSTRAINT_COLLECTION)
    if collection is None:
        collection = bpy.data.collections.new(CONSTRAINT_COLLECTION)
        bpy.context.scene.collection.children.link(collection)
    created = 0
    for control in motion.controls:
        frame = int(control.get("frame", -1))
        if frame < 0:
            fraction = float(control.get("frame_fraction", 0.0))
            frame = int(round(fraction * (motion.n_frames - 1)))
        joints = [int(joint) for joint in control.get("joints") or []]
        values = control.get("values") or []
        stage = str(control.get("stage", ""))
        axes = str(control.get("axes", ""))
        for slot, joint in enumerate(joints):
            row = values[slot] if slot < len(values) else [0.0, 0.0, 0.0]
            world = to_blender_space(np.asarray([row], dtype=np.float32), flip_facing)[0]
            empty = bpy.data.objects.new(_empty_name(frame, joint), None)
            empty.location = Vector(world) * scale
            empty.empty_display_type = "SPHERE"
            empty.empty_display_size = 0.02
            empty["msflow_stage"] = stage
            empty["msflow_axes"] = axes
            empty["msflow_joint"] = motion.joint_names[joint]
            collection.objects.link(empty)
            created += 1
    return created


def clear_constraint_empties() -> int:
    collection = bpy.data.collections.get(CONSTRAINT_COLLECTION)
    if collection is None:
        return 0
    removed = 0
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj)
        removed += 1
    bpy.data.collections.remove(collection)
    return removed


def _empty_name(frame: int, joint: int) -> str:
    return f"MSF_ctrl_f{int(frame):04d}_j{joint:02d}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("run blender_addon/tests/retarget_selftest.py instead")
