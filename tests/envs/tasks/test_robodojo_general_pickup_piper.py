import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import trimesh

from mani_skill.envs.tasks.pick_anything.episode_specs import ObjectSpec
from mani_skill.envs.tasks.pick_anything.general_pickup_specs import (
    CONTRACT_VERSION,
    RoboDojoLayoutSpec,
    load_robodojo_layout_spec,
)
from mani_skill.envs.tasks.pick_anything.randomization.object_sources import (
    RoboDojoConvertedObjectSource,
)
from mani_skill.envs.tasks.pick_anything.robodojo_general_pickup_piper import (
    RoboDojoGeneralPickupPiperEnv,
)
from mani_skill.envs.tasks.tabletop.lift_cube_piper import LIFT_HEIGHT
from mani_skill.utils.registration import REGISTERED_ENVS


def _canonical_bytes(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def _scene_object(*, category, index, label=None, asset_type="Clutter"):
    return {
        "asset_type": asset_type,
        "category": category,
        "category_idx": index,
        "label": label,
        "descriptions": [category] if label == "target" else [],
        "instruction_candidates": (
            [f"Pick up the {category} by 10 cm."] if label == "target" else []
        ),
        "scale": [1.0, 1.0, 1.0],
        "physics": {"type": "rigid", "mass": 0.1, "friction": 0.5},
        "pose_maniskill": {
            "position": [0.0, 0.0, 0.02],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    }


def _contract(clutter_count=1):
    payload = {
        "contract_version": CONTRACT_VERSION,
        "filtered_layout_id": 0,
        "source_layout": {"path": "layout.json", "sha256": "1" * 64},
        "expected_clutter_count": 10,
        "source_clutter_count": clutter_count,
        "source_clutter_count_matches_config": clutter_count == 10,
        "target": _scene_object(
            category="mug", index=2, label="target", asset_type="Rigid"
        ),
        "clutter": [
            _scene_object(category=f"clutter-{index}", index=index)
            for index in range(clutter_count)
        ],
        "fixtures": {},
        "static_contract": {},
    }
    payload["contract_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_v4_asset_root(tmp_path: Path) -> Path:
    asset_root = tmp_path / "assets"
    asset_dir = asset_root / "Rigid" / "test" / "00000"
    asset_dir.mkdir(parents=True)
    visual_path = asset_dir / "visual.glb"
    trimesh.Scene(trimesh.creation.box(extents=[0.04, 0.04, 0.04])).export(visual_path)
    left = trimesh.creation.box(extents=[0.01, 0.02, 0.03])
    left.apply_translation([-0.02, 0.0, 0.0])
    right = left.copy()
    right.apply_translation([0.04, 0.0, 0.0])
    collision = trimesh.util.concatenate([left, right])
    collision_path = asset_dir / "collision.ply"
    collision.export(collision_path)
    conversion = {
        "converter_version": "robodojo_usdz_to_maniskill_v4_material_graph_coacd",
        "asset_key": "Rigid/test/00000",
        "collision_decomposition": {
            "algorithm": "coacd",
            "package_version": "1.0.13",
            "parameters": {"threshold": 0.05, "max_convex_hull": 32, "seed": 0},
        },
        "collision_hull_count": 2,
        "collision_vertices": len(collision.vertices),
        "collision_triangles": len(collision.faces),
        "collision_volume": float(abs(left.volume) + abs(right.volume)),
        "visual.glb_sha256": _sha256_file(visual_path),
        "collision.ply_sha256": _sha256_file(collision_path),
    }
    (asset_dir / "conversion.json").write_bytes(_canonical_bytes(conversion))
    manifest = {"assets": [conversion]}
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    (asset_root / "manifest.json").write_bytes(_canonical_bytes(manifest))
    return asset_root


def test_robodojo_object_spec_requires_category() -> None:
    with pytest.raises(ValueError, match="robodojo ObjectSpec requires category"):
        ObjectSpec(source="robodojo", object_id="00000")

    spec = ObjectSpec(source="robodojo", category="mug", object_id="00002")
    assert spec.stable_id == "robodojo/mug/00002"


def test_layout_contract_accepts_source_clutter_count_mismatch() -> None:
    spec = RoboDojoLayoutSpec.from_dict(_contract(clutter_count=9))

    assert len(spec.clutter) == 9
    assert spec.expected_clutter_count == 10
    assert spec.prompt == "Pick up the mug by 10 cm."
    assert spec.target.object_spec.stable_id == "robodojo/mug/00002"


def test_layout_contract_rejects_tampering() -> None:
    payload = _contract()
    payload["target"]["category"] = "changed"

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        RoboDojoLayoutSpec.from_dict(payload)


def test_layout_loader_binds_contract_and_conversion_manifests(tmp_path) -> None:
    contract_root = tmp_path / "contracts"
    asset_root = tmp_path / "assets"
    contract_root.mkdir()
    asset_root.mkdir()
    contract = _contract()
    contract_path = contract_root / "general_pickup_filtered_000.json"
    contract_path.write_bytes(_canonical_bytes(contract))
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "assets_root": "/source/assets",
        "piper_assets_root": "/source/piper",
        "seed": 0,
        "layout_ids": [0],
        "contracts": [
            {
                "filtered_layout_id": 0,
                "contract_file": contract_path.name,
                "contract_sha256": contract["contract_sha256"],
            }
        ],
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    manifest_path = contract_root / "manifest.json"
    manifest_path.write_bytes(_canonical_bytes(manifest))
    conversion_manifest = {
        "converter_version": "robodojo_usdz_to_maniskill_v2",
        "contract_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "asset_count": 0,
        "assets": [],
    }
    conversion_manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_bytes(conversion_manifest)
    ).hexdigest()
    (asset_root / "manifest.json").write_bytes(_canonical_bytes(conversion_manifest))

    spec = load_robodojo_layout_spec(
        layout_id=0, contract_root=contract_root, asset_root=asset_root
    )

    assert spec.filtered_layout_id == 0


def test_layout_loader_binds_generated_contract_to_conversion_manifest(
    tmp_path,
) -> None:
    contract_root = tmp_path / "contracts"
    asset_root = tmp_path / "assets"
    contract_root.mkdir()
    asset_root.mkdir()
    contract = _contract()
    contract_path = contract_root / "general_pickup_filtered_000.json"
    contract_path.write_bytes(_canonical_bytes(contract))
    conversion_manifest = {
        "converter_version": "robodojo_usdz_to_maniskill_v3_material_graph",
        "contract_manifest_sha256": "0" * 64,
        "asset_count": 0,
        "assets": [],
    }
    conversion_manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_bytes(conversion_manifest)
    ).hexdigest()
    conversion_manifest_path = asset_root / "manifest.json"
    conversion_manifest_path.write_bytes(_canonical_bytes(conversion_manifest))
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "profile": "mirror_train",
        "conversion_manifest_sha256": hashlib.sha256(
            conversion_manifest_path.read_bytes()
        ).hexdigest(),
        "layout_ids": [0],
        "contracts": [
            {
                "filtered_layout_id": 0,
                "contract_file": contract_path.name,
                "contract_sha256": contract["contract_sha256"],
            }
        ],
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    manifest_path = contract_root / "manifest.json"
    manifest_path.write_bytes(_canonical_bytes(manifest))

    spec = load_robodojo_layout_spec(
        layout_id=0, contract_root=contract_root, asset_root=asset_root
    )

    assert spec.filtered_layout_id == 0
    conversion_manifest["converter_version"] = "tampered"
    conversion_manifest_path.write_bytes(_canonical_bytes(conversion_manifest))
    with pytest.raises(RuntimeError, match="different conversion manifest"):
        load_robodojo_layout_spec(
            layout_id=0, contract_root=contract_root, asset_root=asset_root
        )


def test_environment_registration_and_control_timing() -> None:
    assert REGISTERED_ENVS["RoboDojoGeneralPickupPiper-v1"].max_episode_steps == 200
    config = RoboDojoGeneralPickupPiperEnv._default_sim_config.fget(
        object.__new__(RoboDojoGeneralPickupPiperEnv)
    )
    assert config.sim_freq == 200
    assert config.control_freq == 20


def test_robodojo_converted_source_accepts_material_graph_converter() -> None:
    assert RoboDojoConvertedObjectSource.SUPPORTED_CONVERTER_VERSIONS == {
        "robodojo_usdz_to_maniskill_v2",
        "robodojo_usdz_to_maniskill_v3_material_graph",
        "robodojo_usdz_to_maniskill_v4_material_graph_coacd",
    }


def test_v4_robodojo_source_uses_predecomposed_collision_components(tmp_path) -> None:
    source = RoboDojoConvertedObjectSource(
        asset_type="Rigid",
        category="test",
        object_id="00000",
        asset_root=_write_v4_asset_root(tmp_path),
    )
    calls = []

    class _Builder:
        initial_pose = None

        def add_multiple_convex_collisions_from_file(self, **kwargs):
            calls.append(("collision", kwargs))

        def add_visual_from_file(self, **kwargs):
            calls.append(("visual", kwargs))

        def set_scene_idxs(self, scene_idxs):
            calls.append(("scene_idxs", scene_idxs))

        def build(self, name):
            calls.append(("build", name))
            return name

    env = SimpleNamespace(
        scene=SimpleNamespace(create_actor_builder=lambda: _Builder())
    )

    source.build_actor(env, 0, None)

    collision_call = next(value for name, value in calls if name == "collision")
    assert collision_call["filename"].endswith("collision.ply")
    assert collision_call["decomposition"] == "none"


def test_robodojo_height_metric_does_not_end_episode_before_robust_hold() -> None:
    env = object.__new__(RoboDojoGeneralPickupPiperEnv)
    env.object_rest_z = torch.tensor([0.02])
    env.obj = SimpleNamespace(
        pose=SimpleNamespace(p=torch.tensor([[0.0, 0.0, 0.02 + LIFT_HEIGHT + 1e-3]]))
    )
    env.agent = SimpleNamespace(is_grasping=lambda obj: torch.tensor([False]))
    env.success_streak = torch.tensor([9], dtype=torch.int32)
    env.success_streak_steps = 10
    env.streak_updated_at = torch.tensor([9], dtype=torch.int32)
    env._elapsed_steps = torch.tensor([10], dtype=torch.int32)

    info = env.evaluate()

    assert bool(info["robodojo_success"][0])
    assert not bool(info["success"][0])
    assert not bool(info["robust_success_10step"][0])
