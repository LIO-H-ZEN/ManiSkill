from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest
import trimesh
import yaml

from mani_skill.envs.tasks.pick_anything.episode_specs import ObjectSpec
from mani_skill.examples.motionplanning.piper.grasping.contracts import (
    GraspCandidate,
    GraspProviderName,
)
from mani_skill.examples.motionplanning.piper.grasping.geometry import (
    ResolvedObjectGeometry,
)
from mani_skill.examples.motionplanning.piper.grasping.graspgenx import (
    GRASPGENX_GENERATION_CONTRACT_VERSION,
    GraspGenXConfig,
    GraspGenXGraspProvider,
)
from mani_skill.examples.motionplanning.piper.grasping.gripper_geometry import (
    PiperGripperGeometry,
)


def _geometry() -> ResolvedObjectGeometry:
    mesh = trimesh.creation.box(extents=[0.02, 0.04, 0.06])
    return ResolvedObjectGeometry(
        object_spec=ObjectSpec(
            source="cube",
            object_id="cube",
            cube_half_size=0.03,
            cube_color=(1.0, 0.0, 0.0, 1.0),
        ),
        proposal_mesh=mesh,
        collision_meshes=(mesh.copy(),),
        object_T_mesh=np.eye(4),
        source_files=(),
        collision_source="primitive",
        canonical_geometry_hash="a" * 64,
    )


def _write_yaml(path, grasps):
    path.write_text(
        yaml.safe_dump(
            {"format": "isaac_grasp", "format_version": 1.0, "grasps": grasps}
        )
    )


def test_graspgenx_converts_hand_frame_to_piper_tcp(tmp_path):
    source = tmp_path / "grasps.yml"
    _write_yaml(
        source,
        {
            "low": {
                "confidence": 0.2,
                "position": [0.1, 0.2, 0.3],
                "orientation": {"w": 1.0, "xyz": [0.0, 0.0, 0.0]},
            },
            "high": {
                "confidence": 0.9,
                "position": [0.0, 0.0, 0.0],
                "orientation": {"w": 1.0, "xyz": [0.0, 0.0, 0.0]},
            },
        },
    )

    candidates = GraspGenXGraspProvider(
        GraspGenXConfig(
            maximum_candidates=2,
            generation_num_grasps=8,
            generation_topk_num_grasps=2,
        )
    ).load_isaac_yaml(source, _geometry())

    assert [item.metadata["source_candidate_id"] for item in candidates] == [
        "high",
        "low",
    ]
    candidate = candidates[0]
    assert candidate.source is GraspProviderName.GRASPGENX
    np.testing.assert_allclose(candidate.object_T_tcp[:3, 3], [0.0, 0.0, 0.1358])
    np.testing.assert_allclose(
        candidate.object_T_tcp[:3, :3],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    np.testing.assert_allclose(candidate.object_T_tcp[:3, 2], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(candidate.object_T_tcp[:3, 1], [-1.0, 0.0, 0.0])
    assert candidate.required_width == pytest.approx(0.001)
    json.dumps(candidate.metadata)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"format": "wrong", "format_version": 1.0, "grasps": {}},
        {"format": "isaac_grasp", "format_version": 1.0, "grasps": {}},
        {
            "format": "isaac_grasp",
            "format_version": 1.0,
            "grasps": {
                "bad": {
                    "confidence": float("nan"),
                    "position": [0.0, 0.0, 0.0],
                    "orientation": {"w": 1.0, "xyz": [0.0, 0.0, 0.0]},
                }
            },
        },
    ],
)
def test_graspgenx_malformed_output_fails_fast(tmp_path, payload):
    source = tmp_path / "grasps.yml"
    source.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError):
        GraspGenXGraspProvider().load_isaac_yaml(source, _geometry())


def test_graspgenx_cache_fingerprint_tracks_collision_geometry():
    provider = GraspGenXGraspProvider(GraspGenXConfig())
    first = _geometry()
    second_mesh = first.collision_meshes[0].copy()
    second_mesh.vertices[0, 0] += 0.001
    second = dataclasses.replace(first, collision_meshes=(second_mesh,))
    assert provider.cache_fingerprint(first) != provider.cache_fingerprint(second)


def test_graspgenx_fingerprint_tracks_generation_contract() -> None:
    first = GraspGenXConfig()
    second = dataclasses.replace(first, generation_seed=1)
    assert first.fingerprint != second.fingerprint


def test_graspgenx_rejects_wrong_raw_candidate_count(tmp_path) -> None:
    source = tmp_path / "grasps.yml"
    _write_yaml(
        source,
        {
            "only": {
                "confidence": 0.9,
                "position": [0.0, 0.0, 0.0],
                "orientation": {"w": 1.0, "xyz": [0.0, 0.0, 0.0]},
            }
        },
    )
    with pytest.raises(ValueError, match="generation contract"):
        GraspGenXGraspProvider().load_isaac_yaml(source, _geometry())


def test_graspgenx_validates_deterministic_generation_manifest(tmp_path) -> None:
    source = tmp_path / "grasps.yml"
    source.write_text("grasps: {}\n")
    config = GraspGenXConfig()
    manifest = {
        "contract_version": GRASPGENX_GENERATION_CONTRACT_VERSION,
        "implementation_revision": config.implementation_revision,
        "checkpoint_revision": config.checkpoint_revision,
        "gripper_assets_revision": config.gripper_assets_revision,
        "gripper_name": config.gripper_name,
        "seed": config.generation_seed,
        "num_grasps": config.generation_num_grasps,
        "topk_num_grasps": config.generation_topk_num_grasps,
        "num_sample_points": config.generation_num_sample_points,
        "grasp_threshold": config.generation_grasp_threshold,
        "output_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    source.with_suffix(".yml.generation.json").write_text(json.dumps(manifest))

    loaded = GraspGenXGraspProvider(config).validate_generation_manifest(source)

    assert loaded == manifest


def test_graspgenx_rejects_generation_manifest_hash_mismatch(tmp_path) -> None:
    source = tmp_path / "grasps.yml"
    source.write_text("grasps: {}\n")
    config = GraspGenXConfig()
    manifest = {
        "contract_version": GRASPGENX_GENERATION_CONTRACT_VERSION,
        "implementation_revision": config.implementation_revision,
        "checkpoint_revision": config.checkpoint_revision,
        "gripper_assets_revision": config.gripper_assets_revision,
        "gripper_name": config.gripper_name,
        "seed": config.generation_seed,
        "num_grasps": config.generation_num_grasps,
        "topk_num_grasps": config.generation_topk_num_grasps,
        "num_sample_points": config.generation_num_sample_points,
        "grasp_threshold": config.generation_grasp_threshold,
        "output_sha256": "0" * 64,
    }
    source.with_suffix(".yml.generation.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="SHA-256"):
        GraspGenXGraspProvider(config).validate_generation_manifest(source)


def test_graspgenx_refines_contact_width_and_closing_axis() -> None:
    collision = trimesh.creation.box(extents=[0.02, 0.03, 0.04])
    collision.apply_translation([0.0, 0.004, -0.02])
    geometry = dataclasses.replace(
        _geometry(),
        collision_meshes=(collision,),
        canonical_geometry_hash="b" * 64,
    )
    raw = GraspCandidate(
        candidate_id="raw",
        source=GraspProviderName.GRASPGENX,
        object_T_tcp=np.eye(4),
        required_width=0.001,
        proposal_score=0.9,
        contact_points=np.zeros((2, 3)),
    )
    provider = GraspGenXGraspProvider(
        GraspGenXConfig(area_samples=600, balanced_samples=240)
    )

    refined = provider.refine_contact_candidates(
        [raw], geometry, PiperGripperGeometry.from_package_assets()
    )

    assert len(refined) == 1
    candidate = refined[0]
    assert candidate.object_T_tcp[1, 3] == pytest.approx(0.004, abs=5e-4)
    assert candidate.required_width == pytest.approx(0.0303, abs=1e-3)
    assert candidate.metadata["contact_refined"] is True
    assert candidate.metadata["contact_surface"] == "primitive"
