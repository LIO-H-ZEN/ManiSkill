import hashlib
import math

import pytest

from mani_skill.envs.tasks.pick_anything.general_pickup_specs import CONTRACT_VERSION
from mani_skill.envs.tasks.pick_anything.general_pickup_training_contracts import (
    PIPER_BASE_EXCLUSION_RADIUS,
    PIPER_BASE_XY,
    TRAINING_CONTRACT_GENERATOR_VERSION,
    canonical_json_bytes,
    generate_layout,
)


def _scene_object(*, asset_type: str, category: str, index: int, radius: float):
    label = "target" if asset_type == "Rigid" else None
    return {
        "asset_type": asset_type,
        "category": category,
        "category_idx": index,
        "label": label,
        "descriptions": [category] if label else [],
        "instruction_candidates": (
            [f"Pick up the {category} by 10 cm."] if label else []
        ),
        "scale": [1.0, 1.0, 1.0],
        "physics": {"type": "rigid", "mass": 0.1, "friction": 0.5},
        "metadata": {
            "active": {"place": {"up": {"projection_circle": {"radius": radius}}}},
            "geometry": {"oriented_bbox": {"extents": [0.02, 0.02, 0.02]}},
        },
        "pose_maniskill": {
            "position": [0.0, 0.0, 0.01],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "pose_robodojo": {
            "position": [0.0, -0.1, 0.775],
            "orientation_wxyz": [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
        },
    }


def test_generate_layout_is_deterministic_and_collision_free() -> None:
    target = _scene_object(asset_type="Rigid", category="target", index=0, radius=0.025)
    clutter = [
        _scene_object(
            asset_type="Clutter", category=f"clutter-{index}", index=0, radius=0.02
        )
        for index in range(10)
    ]
    kwargs = {
        "layout_id": 1000,
        "generation_seed": 1234,
        "target_source": target,
        "clutter_sources": clutter,
        "template": {"fixtures": {"table": "wood"}, "static_contract": {}},
        "margin": 0.02,
    }

    first = generate_layout(**kwargs)
    second = generate_layout(**kwargs)

    assert first == second
    assert first["profile"] == "mirror_train"
    assert first["generator_version"] == TRAINING_CONTRACT_GENERATOR_VERSION
    assert first["source_layout"]["path"] == "generated://mirror_train/001000"
    assert first["target"]["instruction_candidates"] == ["Pick up the target by 10 cm."]
    assert len({item["category"] for item in first["clutter"]}) == 10
    payload = dict(first)
    stored_hash = payload.pop("contract_sha256")
    assert stored_hash == hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    objects = [first["target"], *first["clutter"]]
    circles = []
    for item in objects:
        x, y, _ = item["pose_maniskill"]["position"]
        radius = item["metadata"]["active"]["place"]["up"]["projection_circle"][
            "radius"
        ]
        circles.append((x, y, radius))
        rx, ry, rz = item["pose_robodojo"]["position"]
        assert rx == pytest.approx(-y)
        assert ry == pytest.approx(x - 0.1)
        assert rz == pytest.approx(item["pose_maniskill"]["position"][2] + 0.765)
    for index, (x, y, radius) in enumerate(circles):
        assert math.hypot(x - PIPER_BASE_XY[0], y - PIPER_BASE_XY[1]) >= (
            radius + PIPER_BASE_EXCLUSION_RADIUS + 0.02 - 1e-12
        )
        for other_x, other_y, other_radius in circles[index + 1 :]:
            assert math.hypot(x - other_x, y - other_y) >= (
                radius + other_radius + 0.02 - 1e-12
            )


def test_generate_layout_requires_ten_clutter_objects() -> None:
    target = _scene_object(asset_type="Rigid", category="target", index=0, radius=0.025)
    with pytest.raises(ValueError, match="exactly 10"):
        generate_layout(
            layout_id=1000,
            generation_seed=1234,
            target_source=target,
            clutter_sources=[],
            template={},
            margin=0.02,
        )


def test_generate_layout_honors_reachable_target_bounds() -> None:
    target = _scene_object(asset_type="Rigid", category="target", index=0, radius=0.025)
    clutter = [
        _scene_object(
            asset_type="Clutter", category=f"clutter-{index}", index=0, radius=0.02
        )
        for index in range(10)
    ]
    layout = generate_layout(
        layout_id=2000,
        generation_seed=9876,
        target_source=target,
        clutter_sources=clutter,
        template={"fixtures": {}, "static_contract": {}},
        margin=0.02,
        target_xlim=(-0.08, 0.02),
        target_ylim=(-0.13, -0.04),
    )

    x, y, _ = layout["target"]["pose_maniskill"]["position"]
    assert -0.08 <= x <= 0.02
    assert -0.13 <= y <= -0.04


def test_generate_layout_rejects_invalid_target_bounds() -> None:
    target = _scene_object(asset_type="Rigid", category="target", index=0, radius=0.025)
    clutter = [
        _scene_object(
            asset_type="Clutter", category=f"clutter-{index}", index=0, radius=0.02
        )
        for index in range(10)
    ]
    with pytest.raises(ValueError, match="invalid target_xlim"):
        generate_layout(
            layout_id=2000,
            generation_seed=9876,
            target_source=target,
            clutter_sources=clutter,
            template={},
            margin=0.02,
            target_xlim=(0.02, -0.08),
        )


def test_generated_contract_uses_supported_schema() -> None:
    assert CONTRACT_VERSION == "robodojo_general_pickup_piper_mvp_v1"
    assert TRAINING_CONTRACT_GENERATOR_VERSION.startswith("robodojo_mirror_train_")
