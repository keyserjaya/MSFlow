"""Map the 22 MSFlow joints onto a target rig's bones by name patterns.

Pure python (no bpy). The matcher normalizes bone names (`DEF-thigh.L`,
`mixamorig:LeftUpLeg`, `upperarm_l` compare equal), scores them against
per-role signature lists (priority-ordered), and assigns each target bone at
most once. Mechanical bones (MCH/WGT) are excluded so animation lands on
deform/control bones. Slots are requested in an order that resolves the
conventional ambiguities (shins before thighs, Mixamo's neutral `Leg`,
clavicles before upper arms, spines by ascending digits).
"""
from __future__ import annotations

from typing import Iterable, Sequence

# HumanML3D t2m joint order (mirror of webui/skeleton.py in the MSFlow repo).
JOINT_NAMES: tuple[str, ...] = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)

ROLE_BY_JOINT: dict[int, str] = {
    0: "pelvis",
    1: "thigh", 2: "thigh",
    3: "torso_low", 6: "torso_mid", 9: "torso_high",
    4: "shin", 5: "shin",
    7: "foot", 8: "foot",
    10: "toe", 11: "toe",
    12: "neck",
    13: "clavicle", 14: "clavicle",
    15: "head",
    16: "upperarm", 17: "upperarm",
    18: "forearm", 19: "forearm",
    20: "hand", 21: "hand",
}

SIDE_BY_JOINT: dict[int, str] = {
    1: "left", 2: "right", 4: "left", 5: "right", 7: "left", 8: "right",
    10: "left", 11: "right", 13: "left", 14: "right", 16: "left", 17: "right",
    18: "left", 19: "right", 20: "left", 21: "right",
}

# Greedy request order: scarce/specific anatomy first so plain names cannot
# poach what a neighbouring slot needs (Mixamo names shins `Leg`, etc.).
DEFAULT_JOINT_ORDER: tuple[int, ...] = (
    0, 7, 8, 10, 11, 4, 5, 1, 2, 13, 14, 16, 17, 18, 19, 20, 21, 12, 15,
)

# Per-role signature tuples; every substring must appear in the normalized
# name and earlier tuples win.
ROLE_SIGNATURES: dict[str, tuple[tuple[str, ...], ...]] = {
    "pelvis": (("hips",), ("pelvis",), ("root",)),
    "foot": (("foot",), ("ankle",)),
    "toe": (("toe",), ("ball",), ("foot", "end")),
    "shin": (("lowerleg",), ("lower", "leg"), ("shin",), ("calf",), ("knee",), ("leg",)),
    "thigh": (("thigh",), ("upperleg",), ("upleg",), ("upper", "leg"), ("leg",)),
    "neck": (("neck",),),
    "head": (("head",),),
    "clavicle": (("clavicle",), ("collar",), ("shoulder",)),
    "upperarm": (("upperarm",), ("upper", "arm"), ("bicep",), ("arm",)),
    "forearm": (("forearm",), ("fore", "arm"), ("lowerarm",), ("lower", "arm"), ("elbow",)),
    "hand": (("hand",), ("wrist",)),
}

# Token must be absent for the role even when the signature matches.
ROLE_REJECT: dict[str, tuple[str, ...]] = {
    "pelvis": (),
    "foot": ("toe", "ball", "end", "rolling", "twist"),
    "toe": (),  # foot-bone rejection is special-cased below
    "shin": ("thigh", "twist"),
    "thigh": ("shin", "calf", "lower", "knee", "twist"),
    "neck": ("twist",),
    "head": ("eye", "jaw", "cheek", "teeth", "ear", "nose", "lid", "tongue"),
    "clavicle": ("twist",),
    "upperarm": ("fore", "lower", "twist"),
    "forearm": ("upper", "twist"),
    "hand": ("thumb", "finger", "index", "middle", "ring", "pinky", "tip"),
}

REQUIRED_JOINTS = (0, 1, 2, 4, 5, 7, 8, 3, 16, 17)


def normalize(name: str) -> str:
    lowered = name.strip().lower()
    for char in (":", "-", " ", "."):
        lowered = lowered.replace(char, "_")
    while "__" in lowered:
        lowered = lowered.replace("__", "_")
    return lowered.strip("_")


def _is_mechanical(norm: str) -> bool:
    return (
        norm.startswith(("mch", "wgt", "mesh"))
        or "_mch" in norm
        or "_ik_" in norm
        or norm.endswith(("_ik", "_pole"))
    )


def _candidate_order(norm: str) -> tuple[int, int, str]:
    """Prefer plain/deform over org/ik-ish names, then compact names."""
    fk_or_ik = 1 if (norm.endswith("_fk") or "_ik" in norm) else 0
    org = 1 if norm.startswith("org_") else 0
    return (org, fk_or_ik, norm)


def has_side_token(norm: str, side: str) -> bool:
    parts = norm.split("_")
    if side == "left":
        return "left" in norm or "l" in parts
    return "right" in norm or "r" in parts


def _is_side_mismatch(norm: str, side: str) -> bool:
    left = has_side_token(norm, "left")
    right = has_side_token(norm, "right")
    if side == "left":
        return not left and right
    if side == "right":
        return not right and left
    return left != right


def _rejects(norm: str, role: str) -> bool:
    if role == "toe":
        # a bare foot bone must not consume the toe slot
        return "foot" in norm and "ball" not in norm and "end" not in norm
    return any(token in norm for token in ROLE_REJECT.get(role, ()))


def resolve_torso_slots(candidates: Sequence[str]) -> dict[int, str]:
    """Order spine/chest-like bones into torso slots 3, 6, 9: digit-free
    names first (`spine` before `spine1`), ascending digits, then chest names,
    with `upper`-prefixed names last."""
    ranked: list[tuple[tuple[int, int, int, str], str]] = []
    for name in candidates:
        norm = normalize(name)
        if not any(token in norm for token in ("spine", "chest", "torso")):
            continue
        tail = norm[len(norm.rstrip("0123456789")):]
        digits = int(tail) if tail else 0
        family = 0 if "spine" in norm else 1
        upper = 1 if "upper" in norm else 0
        ranked.append(((family, digits, upper, norm), name))
    ranked.sort(key=lambda item: item[0])
    return {joint: name for joint, (_, name) in zip((3, 6, 9), ranked)}


def auto_match(
    bone_names: Iterable[str],
    joints: Sequence[int] = tuple(range(22)),
) -> dict[int, str]:
    """Assign target bone names to the requested joint ids; unspecified
    joints stay absent. Returns {joint_id: bone_name}."""
    candidates: list[str] = sorted(
        {
            name
            for name in bone_names
            if name and not _is_mechanical(normalize(name))
        },
        key=lambda name: _candidate_order(normalize(name)),
    )
    used: set[str] = set()
    mapping: dict[int, str] = {}

    torso = [joint for joint in joints if joint in (3, 6, 9)]
    for joint in [j for j in DEFAULT_JOINT_ORDER if j in joints and j not in torso]:
        role = ROLE_BY_JOINT[joint]
        side = SIDE_BY_JOINT.get(joint, "center")
        for signature in ROLE_SIGNATURES[role]:
            for name in candidates:
                if name in used:
                    continue
                norm = normalize(name)
                if _is_side_mismatch(norm, side) or _rejects(norm, role):
                    continue
                if all(token in norm for token in signature):
                    mapping[joint] = name
                    used.add(name)
                    break
            if joint in mapping:
                break

    torso_candidates = [name for name in candidates if name not in used]
    for joint, name in resolve_torso_slots(torso_candidates).items():
        if joint in torso:
            mapping[joint] = name
            used.add(name)
    return mapping


def missing_required(mapping: dict[int, str]) -> list[int]:
    return [joint for joint in REQUIRED_JOINTS if joint not in mapping]


def sanitize_preset(raw: object) -> dict[int, str]:
    """Validate/convert a mapping-preset JSON payload ({joint: bone})."""
    if not isinstance(raw, dict):
        raise ValueError("mapping preset must be a JSON object")
    mapping: dict[int, str] = {}
    for key, value in raw.items():
        try:
            joint = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"mapping preset key must be a joint id: {key!r}") from exc
        if not 0 <= joint < 22 or not isinstance(value, str):
            raise ValueError(f"mapping preset entry invalid: {key!r}: {value!r}")
        mapping[joint] = value
    return mapping


def preset_doc(mapping: dict[int, str]) -> dict[str, str]:
    return {str(joint): mapping[joint] for joint in sorted(mapping)}


def joint_field_name(joint: int) -> str:
    return f"map_{joint:02d}"


def joint_field_label(name: str, joint: int) -> str:
    return f"{joint:02d} {name}"
