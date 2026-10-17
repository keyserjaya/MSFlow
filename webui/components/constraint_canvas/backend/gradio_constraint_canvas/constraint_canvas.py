"""Editable 3D constraint scene (Unity-style) for MSFlow motion constraints.

The python-level value is a plain dict:

    {
        "frame": int,  # current editing frame
        "underlay": {"points": [{"joint", "x", "y", "z"}], "edges": [[i, j]]},
        "markers": [  # one handle per saved constraint, any frame
            {
                "id": "f{frame}-j{joint}",
                "joint": int,
                "frame": int,
                "x": float, "y": float, "z": float,
                "axes": ["x", ...],       # constrained position axes
                "color": str, "controlled": bool, "selected": bool,
            }
        ],
        "lines": [{"points": [[x, y, z], ...], "color"}],
        "selected_joint": int | None,
        "label": str,
        "playback": {"frames": [[[x, y, z]]], "fps": int, "label": str} | None,
    }

Handles at `frame == value.frame` are solid and gizmo-able; other frames'
handles render as ghosts (clicking one jumps the editing frame). The
frontend IK-solves the armature (underlay) onto the current frame's handles.

Change events emit plain dicts (passed through preprocess unmodified):

    {"action": "move", "joint": id, "x": float, "y": float, "z": float,
     "axes": ["x", ...]}          # gizmo drag commit; axes touched by the drag
    {"action": "select", "joint": id, "frame"?: int}
                                 # -1 deselects; frame set = ghost jump
    {"action": "snapshot", "frame": f, "pose": [[x, y, z], ...22]}
                                 # pin the solved pose at frame f ("constrain
                                 # this frame" button)
"""

from __future__ import annotations

from gradio.components.base import Component
from gradio.events import Events


class ConstraintCanvas(Component):
    """Renders draggable joint constraint handles over an IK-solved armature."""

    EVENTS = [Events.change]

    def preprocess(self, payload):
        """Parameters:
            payload: the event payload / value sent from the frontend
        Returns:
            the payload dict passed to the user's function untouched
        """
        return payload

    def postprocess(self, value):
        """Parameters:
            value: the canvas state dict from the backend
        Returns:
            the same dict sent to the frontend untouched
        """
        return value

    def example_payload(self):
        return {
            "action": "move",
            "joint": 21,
            "x": 0.1,
            "y": 0.9,
            "z": -0.2,
            "axes": ["x", "z"],
        }

    def example_value(self):
        return {
            "frame": 0,
            "underlay": {
                "points": [{"joint": 0, "x": 0.0, "y": 0.95, "z": 0.0}],
                "edges": [],
            },
            "markers": [],
            "lines": [],
            "selected_joint": None,
            "label": "",
        }

    def api_info(self):
        return {
            "type": {},
            "description": "constraint canvas state (free-form json object)",
        }
