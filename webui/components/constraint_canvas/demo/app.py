import gradio as gr
from gradio_constraint_canvas import ConstraintCanvas

EDGES = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 4],
    [4, 7],
    [7, 10],
    [2, 5],
    [5, 8],
    [8, 11],
    [3, 6],
    [6, 9],
    [9, 12],
    [12, 15],
    [12, 13],
    [12, 14],
    [13, 16],
    [16, 18],
    [18, 20],
    [14, 17],
    [17, 19],
    [19, 21],
]
UNDERLAY = {
    "points": [
        {"joint": j, "x": 0.02 * (j % 3 - 1), "y": 0.4 + 0.03 * j, "z": 0.01 * j}
        for j in range(22)
    ],
    "edges": EDGES,
}
VALUE = {
    "frame": 0,
    "underlay": UNDERLAY,
    "markers": [
        {
            "id": "f0-j21",
            "joint": 21,
            "frame": 0,
            "x": 0.12,
            "y": 1.0,
            "z": 0.30,
            "axes": ["x", "z"],
            "color": "#CC79A7",
            "controlled": True,
            "selected": True,
        },
        {
            "id": "f12-j21",
            "joint": 21,
            "frame": 12,
            "x": -0.10,
            "y": 0.95,
            "z": -0.20,
            "axes": ["x", "z"],
            "color": "#CC79A7",
            "controlled": True,
            "selected": False,
        },
    ],
    "lines": [
        {"points": [[0.12, 0.30], [0.12, 0.02], [0.12, -0.28]], "color": "#CC79A7"}
    ],
    "selected_joint": 21,
    "label": "demo: solid handle (f0) + ghost handle (f12)",
}


def show(payload: dict) -> str:
    return str(payload)


with gr.Blocks() as demo:
    gr.Markdown("# ConstraintCanvas demo\nDrag the marker, click to select.")
    canvas = ConstraintCanvas(
        value=VALUE, label="constraint canvas", info="XZ top view"
    )
    payload_view = gr.JSON(label="last event payload")
    canvas.change(show, inputs=[canvas], outputs=[payload_view])


if __name__ == "__main__":
    demo.launch()
