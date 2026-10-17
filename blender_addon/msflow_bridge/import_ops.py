"""Persistent settings + operators for the MSFlow Blender bridge."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Object, Operator, PropertyGroup, Scene

from . import io_msflow, mapping, retarget, server


def _poll_armature(_self: Any, obj: Object) -> bool:
    return obj.type == "ARMATURE"


class MSFlowBridgeSettings(PropertyGroup):
    target: PointerProperty(  # type: ignore[arg-type]
        name="Target armature", type=Object, poll=_poll_armature
    )
    root_motion: EnumProperty(
        name="Root motion",
        items=(
            ("hips", "On hips bone", "Keyframe the mapped pelvis/hips bone"),
            ("object", "On armature object", "Leave the pelvis bone; move the whole object"),
            ("in_place", "In place", "Strip horizontal drift, keep the vertical bob"),
        ),
        default="hips",
    )
    scale_match: BoolProperty(
        name="Match leg scale",
        description="Scale root motion by the thigh+shin span ratio",
        default=True,
    )
    flip_facing: BoolProperty(
        name="Flip facing",
        description="Rotate the motion 180 deg about Z if the character faces the wrong way",
        default=False,
    )
    integer_resample: BoolProperty(
        name="Integer frames",
        description="Round keyframes to whole frames instead of exact fractional timing",
        default=False,
    )
    set_frame_range: BoolProperty(
        name="Set frame range",
        description="Fit the scene playback range to the clip's duration",
        default=True,
    )
    nla_clip: BoolProperty(
        name="Store NLA clip",
        description="Also store the action as a muted 'MSFlow' NLA-track clip "
        "(unmute the track to drive playback from it)",
        default=False,
    )
    constraint_empties: BoolProperty(
        name="Constraint markers",
        description="Import sparse-control targets as empties in 'MSFlow Constraints'",
        default=True,
    )
    out_root: StringProperty(
        name="Run output root",
        description="Folder holding MSFlow run outputs (generations/webui on the shared drive)",
        default="generations/webui",
    )
    receiver_port: IntProperty(name="Receiver port", default=8765, min=1024, max=65535)
    receiver_bind_all: BoolProperty(
        name="Bind all interfaces",
        description="Needed for WSL2 NAT pushes (fires a firewall prompt); "
        "loopback-only otherwise",
        default=False,
    )
    status: StringProperty(name="Status")


for _joint in range(io_msflow.NUM_JOINTS):
    MSFlowBridgeSettings.__annotations__[mapping.joint_field_name(_joint)] = StringProperty(
        name=mapping.joint_field_label(mapping.JOINT_NAMES[_joint], _joint),
        description="Target pose bone for this MSFlow joint (empty = unmapped)",
        default="",
        options={"HIDDEN"},
    )


def settings(context: bpy.types.Context) -> Any:
    return context.scene.msflow_bridge_settings


def read_mapping(context: bpy.types.Context) -> dict[int, str]:
    config = settings(context)
    out: dict[int, str] = {}
    for joint in range(io_msflow.NUM_JOINTS):
        name = getattr(config, mapping.joint_field_name(joint), "").strip()
        if name:
            out[joint] = name
    return out


def write_mapping(context: bpy.types.Context, table: dict[int, str]) -> None:
    config = settings(context)
    for joint in range(io_msflow.NUM_JOINTS):
        setattr(config, mapping.joint_field_name(joint), table.get(joint, ""))


def pick_target(context: bpy.types.Context) -> Object | None:
    config = settings(context)
    if config.target is not None and config.target.type == "ARMATURE":
        return config.target
    active = context.active_object
    if active is not None and active.type == "ARMATURE":
        return active
    armatures = [obj for obj in context.scene.objects if obj.type == "ARMATURE"]
    return armatures[0] if len(armatures) == 1 else None


def options_from_settings(context: bpy.types.Context) -> retarget.RetargetOptions:
    config = settings(context)
    return retarget.RetargetOptions(
        root_motion=str(config.root_motion),
        scale_match=bool(config.scale_match),
        flip_facing=bool(config.flip_facing),
        integer_resample=bool(config.integer_resample),
        set_frame_range=bool(config.set_frame_range),
        nla_clip=bool(config.nla_clip),
        constraint_empties=bool(config.constraint_empties),
        ref_frame=0,
    )


class MSFLOW_OT_import_file(Operator):
    bl_idname = "msflow.import_file"
    bl_label = "Import MSFlow motion"
    bl_description = "Retarget an exported msflow-motion JSON onto the target armature"
    bl_options = {"UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")  # type: ignore[misc]
    filter_glob: StringProperty(  # type: ignore[misc]
        default="*.msflow.json;*.json", options={"HIDDEN"}
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            status = perform_import(context, self.filepath)
        except Exception as exc:
            settings(context).status = f"⛔ import failed: {exc}"
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        settings(context).status = status
        self.report({"INFO"}, status.splitlines()[0])
        return {"FINISHED"}

    def invoke(self, context: bpy.types.Context, _event: Any) -> set[str]:
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


def perform_import(context: bpy.types.Context, path: str) -> str:
    """Load a motion, retarget it onto the armature target, return status."""
    motion = io_msflow.load_motion(path)
    target = pick_target(context)
    if target is None:
        raise ValueError("no armature target - pick one in the MSFlow panel")
    table = read_mapping(context)
    if not table:
        table = mapping.auto_match([bone.name for bone in target.pose.bones])
        write_mapping(context, table)
    report = retarget.retarget_motion(
        motion, target, table, options_from_settings(context), context.scene
    )
    unmatched = [j for j in range(io_msflow.NUM_JOINTS) if j not in table]
    lines = [f"✅ {Path(motion.source).name}: action '{report.action_name}'"]
    lines.append(
        f"{report.frames} frames @ {motion.fps:.0f}fps = {report.duration_sec:.2f}s"
        f" · root scale {report.scale:.3f}"
    )
    if report.mean_error_cm:
        lines.append(
            f"FK deviation: mean {report.mean_error_cm:.2f}cm / max {report.max_error_cm:.2f}cm"
        )
    for warning in report.warnings:
        lines.append(f"⚠ {warning}")
    if unmatched:
        lines.append(
            "unmapped (rest pose kept): "
            + ", ".join(f"{j} {motion.joint_names[j]}" for j in unmatched)
        )
    return "\n".join(lines)


class MSFLOW_OT_import_latest(Operator):
    bl_idname = "msflow.import_latest"
    bl_label = "Import latest run"
    bl_description = "Import every motion from the newest run folder under the output root"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        root = self._resolve_root(context)
        files = io_msflow.find_motion_files(root)
        if not files:
            self.report({"INFO"}, f"no sample_*.msflow.json under {root}")
            settings(context).status = f"⛔ import latest: nothing under {root}"
            return {"CANCELLED"}
        run_files = sorted(files[0].parent.glob("sample_*.msflow.json"))
        notes: list[str] = []
        failed = 0
        for path in run_files:
            try:
                notes.append(perform_import(context, str(path)))
            except Exception as exc:
                failed += 1
                notes.append(f"⛔ {path.name}: {exc}")
        status = "\n".join(notes)
        settings(context).status = status
        self.report({"ERROR" if failed else "INFO"}, notes[0].splitlines()[0])
        return {"CANCELLED" if failed else "FINISHED"}

    @staticmethod
    def _resolve_root(context: bpy.types.Context) -> str:
        raw = str(settings(context).out_root or "")
        if not raw:
            raw = "generations/webui"
        if Path(raw).is_absolute():
            return str(Path(raw))
        blend_path = context.blend_data.filepath
        here = Path(blend_path).parent if blend_path else Path.cwd()
        return str(here / raw)


class MSFLOW_OT_auto_match(Operator):
    bl_idname = "msflow.auto_match"
    bl_label = "Auto-match bones"
    bl_description = "Guess the bone mapping from the target rig's bone names"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        target = pick_target(context)
        if target is None:
            self.report({"ERROR"}, "no armature to scan")
            return {"CANCELLED"}
        table = mapping.auto_match([bone.name for bone in target.pose.bones])
        write_mapping(context, table)
        missing = mapping.missing_required(table)
        line = f"auto-match: {len(table)}/22 joints"
        if missing:
            line += f"; set by hand: {', '.join(map(str, missing))}"
        settings(context).status = line
        self.report({"INFO"}, line)
        return {"FINISHED"}


class MSFLOW_OT_auto_match_geometry(Operator):
    bl_idname = "msflow.auto_match_geometry"
    bl_label = "Match by closest head"
    bl_description = (
        "Assign bones whose rest heads sit closest to the reference pose "
        "(pelvis-aligned + height-scaled) — for rigs with unfamiliar names"
    )
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        target = pick_target(context)
        if target is None:
            self.report({"ERROR"}, "no armature to scan")
            return {"CANCELLED"}
        motion = self._latest_reference(context)
        if motion is None:
            self.report({"ERROR"}, "no msflow motion under the output root")
            return {"CANCELLED"}
        table = self._nearest_heads(context, target, motion)
        write_mapping(context, table)
        line = f"geometry match: {len(table)}/22 joints from {Path(motion.source).name}"
        settings(context).status = line
        self.report({"INFO"}, line)
        return {"FINISHED"}

    def _latest_reference(self, context: bpy.types.Context) -> io_msflow.MotionData | None:
        files = io_msflow.find_motion_files(MSFLOW_OT_import_latest._resolve_root(context))
        return io_msflow.load_motion(files[0]) if files else None

    @staticmethod
    def _hips_bone(target: Object, heads: list[Any]) -> str:
        for pose_bone in target.pose.bones:
            norm = mapping.normalize(pose_bone.name)
            if any(token in norm for token in ("hips", "pelvis", "root")):
                return pose_bone.name
        pairs = [(point.z, pose_bone.name) for point, pose_bone in zip(heads, target.pose.bones)]
        return max(pairs, key=lambda item: item[0])[1]

    @staticmethod
    def _nearest_heads(
        context: bpy.types.Context, target: Object, motion: io_msflow.MotionData
    ) -> dict[int, str]:
        positions = retarget.convert_positions(motion, bool(settings(context).flip_facing))
        reference = [point - positions[0][0] for point in positions[0]]
        source_height = max((point.z for point in reference), default=1.0) or 1.0
        heads = [
            pose_bone.bone.matrix_local.to_translation()
            for pose_bone in target.pose.bones
        ]
        anchor_name = MSFLOW_OT_auto_match_geometry._hips_bone(target, heads)
        anchor_index = next(
            index
            for index, pose_bone in enumerate(target.pose.bones)
            if pose_bone.name == anchor_name
        )
        anchor = heads[anchor_index]
        aligned = [head - anchor for head in heads]
        target_height = max((point.z for point in aligned), default=1.0) or 1.0
        ratio = target_height / source_height if source_height > retarget.EPS else 1.0

        table: dict[int, str] = {}
        used: set[str] = set()
        for joint in range(io_msflow.NUM_JOINTS):
            wanted = reference[joint] * ratio
            best: tuple[float, str] | None = None
            for pose_bone, point in zip(target.pose.bones, aligned):
                if pose_bone.name in used:
                    continue
                distance = (point - wanted).length
                if best is None or distance < best[0]:
                    best = (distance, pose_bone.name)
            if best is not None:
                table[joint] = best[1]
                used.add(best[1])
        return {joint: table[joint] for joint in sorted(table)}


class MSFLOW_OT_save_preset(Operator):
    bl_idname = "msflow.save_mapping_preset"
    bl_label = "Save mapping preset"
    bl_description = "Write the current bone mapping to JSON"
    filepath: StringProperty(subtype="FILE_PATH")  # type: ignore[misc]
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})  # type: ignore[misc]

    def execute(self, context: bpy.types.Context) -> set[str]:
        payload = mapping.preset_doc(read_mapping(context))
        destination = Path(bpy.path.abspath(self.filepath))
        if destination.suffix != ".json":
            destination = destination.with_suffix(".json")
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        settings(context).status = f"mapping preset saved: {destination}"
        self.report({"INFO"}, "mapping preset saved")
        return {"FINISHED"}

    def invoke(self, context: bpy.types.Context, _event: Any) -> set[str]:
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class MSFLOW_OT_load_preset(Operator):
    bl_idname = "msflow.load_mapping_preset"
    bl_label = "Load mapping preset"
    bl_description = "Fill the bone mapping from a saved preset"
    filepath: StringProperty(subtype="FILE_PATH")  # type: ignore[misc]
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})  # type: ignore[misc]

    def execute(self, context: bpy.types.Context) -> set[str]:
        raw = json.loads(Path(bpy.path.abspath(self.filepath)).read_text(encoding="utf-8"))
        table = mapping.sanitize_preset(raw)
        write_mapping(context, table)
        line = f"mapping preset loaded: {len(table)} joints"
        settings(context).status = line
        self.report({"INFO"}, line)
        return {"FINISHED"}

    def invoke(self, context: bpy.types.Context, _event: Any) -> set[str]:
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class MSFLOW_OT_clear_empties(Operator):
    bl_idname = "msflow.clear_empties"
    bl_label = "Clear constraint markers"
    bl_description = "Remove the 'MSFlow Constraints' marker collection"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        removed = retarget.clear_constraint_empties()
        context.scene.msflow_bridge_settings.status = f"cleared {removed} constraint markers"
        self.report({"INFO"}, f"cleared {removed} markers")
        return {"FINISHED"}


WATCH_STATE = {"running": False, "watermark": 0.0}


def watch_tick() -> float | None:
    """bpy.app.timers poller: import runs that appear in the output root."""
    context = bpy.context
    try:
        root = Path(MSFLOW_OT_import_latest._resolve_root(context))
        files = io_msflow.find_motion_files(root)
        for path in files:
            stamp = path.stat().st_mtime
            if stamp <= WATCH_STATE["watermark"]:
                continue
            WATCH_STATE["watermark"] = stamp
            try:
                status = perform_import(context, str(path))
            except Exception as exc:
                status = f"⛔ watch import failed: {exc}"
            settings(context).status = f"watch: {status}"
            break  # one import per tick keeps the UI responsive
    except Exception as exc:
        settings(context).status = f"watch tick failed: {exc}"
    return 2.0 if WATCH_STATE["running"] else None


class MSFLOW_OT_toggle_watch(Operator):
    bl_idname = "msflow.toggle_watch"
    bl_label = "Toggle folder watch"
    bl_description = "Auto-import runs as they land in the output root (WSL2-safe, no port)"

    @classmethod
    def description(cls, _context: bpy.types.Context) -> str:
        return "stop watching" if WATCH_STATE["running"] else "start watching"

    def execute(self, context: bpy.types.Context) -> set[str]:
        if WATCH_STATE["running"]:
            WATCH_STATE["running"] = False
            if bpy.app.timers.is_registered(watch_tick):
                bpy.app.timers.unregister(watch_tick)
            line = "watch stopped"
        else:
            root = MSFLOW_OT_import_latest._resolve_root(context)
            for path in io_msflow.find_motion_files(root):
                WATCH_STATE["watermark"] = max(
                    WATCH_STATE["watermark"], path.stat().st_mtime
                )
            WATCH_STATE["running"] = True
            bpy.app.timers.register(watch_tick, first_interval=0.5, recurring=True)
            line = f"watching {root}"
        settings(context).status = line
        self.report({"INFO"}, line)
        return {"FINISHED"}


class MSFLOW_OT_toggle_receiver(Operator):
    bl_idname = "msflow.toggle_receiver"
    bl_label = "Toggle push receiver"
    bl_description = "Receive 'Send to Blender' pushes from the webUI over HTTP"

    @classmethod
    def description(cls, _context: bpy.types.Context) -> str:
        return "stop listening" if server.RECEIVER.running else "start listening"

    def execute(self, context: bpy.types.Context) -> set[str]:
        config = settings(context)
        if server.RECEIVER.running:
            _stop_receiver()
            line = "receiver stopped"
        else:
            error = server.RECEIVER.start(
                port=int(config.receiver_port),
                bind_all=bool(config.receiver_bind_all),
                callback=lambda path: _pushed_import(path),
            )
            if error:
                settings(context).status = f"⛔ receiver: {error}"
                self.report({"ERROR"}, error)
                return {"CANCELLED"}
            if not bpy.app.timers.is_registered(server.receiver_tick):
                bpy.app.timers.register(
                    server.receiver_tick, first_interval=0.2, recurring=True
                )
            bind = "all interfaces" if config.receiver_bind_all else "127.0.0.1"
            line = f"receiver on http://{bind}:{config.receiver_port}/import"
        settings(context).status = line
        self.report({"INFO"}, line)
        return {"FINISHED"}


def _pushed_import(path: str) -> None:
    import bpy as _bpy

    context = _bpy.context
    try:
        status = perform_import(context, path)
    except Exception as exc:
        status = f"⛔ push import failed: {exc}"
    settings(context).status = status


def _stop_receiver() -> None:
    server.RECEIVER.stop()
    if bpy.app.timers.is_registered(server.receiver_tick):
        bpy.app.timers.unregister(server.receiver_tick)


def shutdown_timers() -> None:
    WATCH_STATE["running"] = False
    _stop_receiver()


CLASSES = (
    MSFlowBridgeSettings,
    MSFLOW_OT_import_file,
    MSFLOW_OT_import_latest,
    MSFLOW_OT_auto_match,
    MSFLOW_OT_auto_match_geometry,
    MSFLOW_OT_save_preset,
    MSFLOW_OT_load_preset,
    MSFLOW_OT_clear_empties,
    MSFLOW_OT_toggle_watch,
    MSFLOW_OT_toggle_receiver,
)


def register_scene_props() -> None:
    bpy.types.Scene.msflow_bridge_settings = PointerProperty(type=MSFlowBridgeSettings)


def unregister_scene_props() -> None:
    del bpy.types.Scene.msflow_bridge_settings
