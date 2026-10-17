
# `gradio_constraint_canvas`
<img alt="Static Badge" src="https://img.shields.io/badge/version%20-%200.0.1%20-%20orange">  

Python library for easily interacting with trained machine learning models

## Installation

```bash
pip install gradio_constraint_canvas
```

## Usage

```python
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

```

## `ConstraintCanvas`

### Initialization

<table>
<thead>
<tr>
<th align="left">name</th>
<th align="left" style="width: 25%;">type</th>
<th align="left">default</th>
<th align="left">description</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><code>value</code></td>
<td align="left" style="width: 25%;">

```python
Any
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>label</code></td>
<td align="left" style="width: 25%;">

```python
str | I18nData | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>info</code></td>
<td align="left" style="width: 25%;">

```python
str | I18nData | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>show_label</code></td>
<td align="left" style="width: 25%;">

```python
bool | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>container</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>value = True</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>scale</code></td>
<td align="left" style="width: 25%;">

```python
int | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>min_width</code></td>
<td align="left" style="width: 25%;">

```python
int | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>interactive</code></td>
<td align="left" style="width: 25%;">

```python
bool | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>visible</code></td>
<td align="left" style="width: 25%;">

```python
bool | Literal['hidden']
```

</td>
<td align="left"><code>value = True</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>elem_id</code></td>
<td align="left" style="width: 25%;">

```python
str | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>elem_classes</code></td>
<td align="left" style="width: 25%;">

```python
list[str] | str | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>render</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>value = True</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>key</code></td>
<td align="left" style="width: 25%;">

```python
int | str | tuple[int | str, ...] | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>preserved_by_key</code></td>
<td align="left" style="width: 25%;">

```python
list[str] | str | None
```

</td>
<td align="left"><code>value = "value"</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>load_fn</code></td>
<td align="left" style="width: 25%;">

```python
Callable | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>every</code></td>
<td align="left" style="width: 25%;">

```python
Timer | float | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>inputs</code></td>
<td align="left" style="width: 25%;">

```python
Component | Sequence[Component] | set[Component] | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>
</tbody></table>


### Events

| name | description |
|:-----|:------------|
| `change` | Triggered when the value of the ConstraintCanvas changes either because of user input (e.g. a user types in a textbox) OR because of a function update (e.g. an image receives a value from the output of an event trigger). See `.input()` for a listener that is only triggered by user input. |



