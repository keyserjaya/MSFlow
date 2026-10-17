"""3D-View N-panel UI for the MSFlow bridge."""
from __future__ import annotations

import bpy
from bpy.types import Panel

from . import io_msflow, mapping
from .import_ops import settings


class MSFLOW_PT_bridge(Panel):
    bl_idname = "MSFLOW_PT_bridge"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MSFlow"
    bl_label = "MSFlow motion bridge"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        config = settings(context)

        row = layout.row(align=True)
        row.prop(config, "target")

        row = layout.row(align=True)
        row.operator("msflow.auto_match", icon="OTHER_OBJECTS")
        row.operator("msflow.auto_match_geometry", text="Closest head", icon="POS_EDGE")
        row = layout.row(align=True)
        row.operator("msflow.load_mapping_preset", text="Load preset")
        row.operator("msflow.save_mapping_preset", text="Save preset")

        box = layout.box()
        box.label(text="Bone mapping (head-anchor joints)", icon="BONE_DATA")
        for joint in range(io_msflow.NUM_JOINTS):
            box.prop(
                config,
                mapping.joint_field_name(joint),
                text=mapping.joint_field_label(mapping.JOINT_NAMES[joint], joint),
            )

        layout.separator()
        row = layout.row(align=True)
        row.operator("msflow.import_file", icon="IMPORT")
        row.operator("msflow.import_latest", text="Latest run", icon="RECOVER_LAST")
        col = layout.column(align=True)
        col.prop(config, "root_motion")
        sub = col.column(align=True)
        sub.prop(config, "scale_match")
        sub.prop(config, "flip_facing")
        sub.prop(config, "integer_resample")
        sub.prop(config, "set_frame_range")
        sub.prop(config, "nla_clip")
        sub.prop(config, "constraint_empties")

        layout.separator()
        col = layout.column(align=True)
        col.operator("msflow.toggle_receiver", icon="RESTRICT_COLOR_ON")
        col.operator("msflow.toggle_watch", icon="VIEWZOOM")
        col.prop(config, "out_root")
        row = col.row(align=True)
        row.prop(config, "receiver_port")
        row.prop(config, "receiver_bind_all", text="bind all")
        row = layout.row(align=True)
        row.operator("msflow.clear_empties", icon="X")

        status = str(config.status or "").strip()
        if status:
            box = layout.box()
            for line in status.splitlines():
                box.label(text=line.strip()[:96])


CLASSES = (MSFLOW_PT_bridge,)
