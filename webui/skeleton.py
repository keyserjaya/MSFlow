"""HumanML3D (t2m) 22-joint layout constants shared by the webui modules."""
from __future__ import annotations

import numpy as np

# HumanML3D t2m joint ordering (verified against sample/joint_control_specs/kick.json,
# which names 0 pelvis, 7 left_ankle, 8 right_ankle, 10/11 feet, 12 neck, 15 head,
# 18/19 elbows, 20/21 wrists).
JOINT_NAMES: dict[int, str] = {
    0: "pelvis",
    1: "left_hip",
    2: "right_hip",
    3: "spine1",
    4: "left_knee",
    5: "right_knee",
    6: "spine2",
    7: "left_ankle",
    8: "right_ankle",
    9: "spine3",
    10: "left_foot",
    11: "right_foot",
    12: "neck",
    13: "left_collar",
    14: "right_collar",
    15: "head",
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
    20: "left_wrist",
    21: "right_wrist",
}

JOINT_CHAIN: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 4),
    (4, 7),
    (7, 10),
    (2, 5),
    (5, 8),
    (8, 11),
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    (12, 13),
    (13, 16),
    (16, 18),
    (18, 20),
    (14, 17),
    (17, 19),
    (19, 21),
)

NUM_JOINTS = 22


def joint_label(joint_id: int) -> str:
    """Human-readable label for a joint id, e.g. '21 right_wrist'."""
    name = JOINT_NAMES.get(int(joint_id), f"joint_{int(joint_id)}")
    return f"{int(joint_id)} {name}"


def canonical_pose() -> np.ndarray:
    """A plausible standing pose [22, 3] in world XYZ (meters), used when no
    HumanML3D reference motion is available.
    """
    body = [
        [0.00, 0.95, 0.00],  # 0 pelvis
        [-0.09, 0.97, 0.00],  # 1 left_hip
        [0.09, 0.97, 0.00],  # 2 right_hip
        [0.00, 1.02, 0.00],  # 3 spine1
        [-0.09, 0.55, 0.02],  # 4 left_knee
        [0.09, 0.55, 0.02],  # 5 right_knee
        [0.00, 1.10, 0.00],  # 6 spine2
        [-0.10, 0.09, 0.05],  # 7 left_ankle
        [0.10, 0.09, 0.05],  # 8 right_ankle
        [0.00, 1.20, 0.00],  # 9 spine3
        [-0.12, 0.03, 0.11],  # 10 left_foot
        [0.12, 0.03, 0.11],  # 11 right_foot
        [0.00, 1.42, 0.02],  # 12 neck
        [-0.06, 1.44, 0.02],  # 13 left_collar
        [0.06, 1.44, 0.02],  # 14 right_collar
        [0.00, 1.62, 0.03],  # 15 head
        [-0.20, 1.40, 0.02],  # 16 left_shoulder
        [0.20, 1.40, 0.02],  # 17 right_shoulder
        [-0.20, 1.12, 0.02],  # 18 left_elbow
        [0.20, 1.12, 0.02],  # 19 right_elbow
        [-0.20, 0.90, 0.05],  # 20 left_wrist
        [0.20, 0.90, 0.05],  # 21 right_wrist
    ]
    pose = np.asarray(body, dtype=np.float32)
    if pose.shape != (NUM_JOINTS, 3):
        raise ValueError("canonical_pose must define 22 XYZ joints")
    return pose
