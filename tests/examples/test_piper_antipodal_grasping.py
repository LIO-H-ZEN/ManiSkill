import pathlib

import numpy as np
import pytest
import trimesh

from mani_skill.envs.tasks.pick_anything.episode_specs import ObjectSpec
from mani_skill.examples.motionplanning.piper.grasping.antipodal import (
    AntipodalConfig,
    AntipodalGraspProvider,
    SurfaceSamples,
    build_contact_pairs,
    sample_surface_mixed,
)
from mani_skill.examples.motionplanning.piper.grasping.cache import GraspCache
from mani_skill.examples.motionplanning.piper.grasping.geometry import (
    ResolvedObjectGeometry,
    resolve_object_geometry,
)
from mani_skill.examples.motionplanning.piper.grasping.oracle import (
    GraspOracleAttempt,
    evaluate_gripper_oracle,
)


def _small_config(**overrides) -> AntipodalConfig:
    values = dict(
        area_samples=192,
        balanced_samples=96,
        maximum_opposites_per_point=3,
        maximum_contact_pairs=64,
        approaches_per_pair=4,
        maximum_candidates=24,
        maximum_per_contact_region=8,
    )
    values.update(overrides)
    return AntipodalConfig(**values)


def test_cube_geometry_matches_actor_scale_and_is_deterministic() -> None:
    spec = ObjectSpec(
        source="cube",
        object_id="cube",
        cube_half_size=0.02,
        cube_color=(1.0, 0.0, 0.0, 1.0),
    )

    first = resolve_object_geometry(spec)
    second = resolve_object_geometry(spec)

    np.testing.assert_allclose(first.proposal_mesh.extents, [0.04, 0.04, 0.04])
    assert first.canonical_geometry_hash == second.canonical_geometry_hash
    assert first.collision_source == "primitive"


def test_mixed_surface_sampling_reserves_balanced_budget() -> None:
    mesh = trimesh.creation.box(extents=[0.04, 0.04, 0.01])
    samples = sample_surface_mixed(
        mesh,
        area_count=60,
        balanced_count=24,
        rng=np.random.default_rng(4),
    )

    assert samples.points.shape == (84, 3)
    assert samples.normals.shape == (84, 3)
    dominant_axes = np.argmax(np.abs(samples.normals), axis=1)
    assert set(dominant_axes.tolist()) == {0, 1, 2}


def test_contact_pairing_is_bounded_and_local_width_aware() -> None:
    samples = SurfaceSamples(
        points=np.array(
            [
                [-0.01, 0.00, 0.00],
                [0.01, 0.00, 0.00],
                [-0.01, 0.03, 0.00],
                [0.01, 0.03, 0.00],
            ]
        ),
        normals=np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        face_indices=np.arange(4),
    )
    config = _small_config(maximum_contact_pairs=2)

    pairs = build_contact_pairs(samples, config)

    assert len(pairs) == 2
    assert all(pair.width == pytest.approx(0.02) for pair in pairs)


def test_antipodal_provider_returns_deterministic_object_local_candidates() -> None:
    spec = ObjectSpec(
        source="cube",
        object_id="cube",
        cube_half_size=0.02,
        cube_color=(1.0, 0.0, 0.0, 1.0),
    )
    geometry = resolve_object_geometry(spec)
    provider = AntipodalGraspProvider(_small_config())

    first = provider.generate(geometry, seed=123)
    second = provider.generate(geometry, seed=123)

    assert len(first) == len(second) == 24
    assert [item.candidate_id for item in first] == [
        item.candidate_id for item in second
    ]
    for left, right in zip(first, second):
        np.testing.assert_allclose(left.object_T_tcp, right.object_T_tcp)
        assert 0.001 <= left.required_width <= 0.068
        assert left.metadata["com_distance"] >= 0.0
        assert left.metadata["gravity_torque_risk"] >= 0.0


def test_grasp_cache_round_trip_and_negative_entry(tmp_path: pathlib.Path) -> None:
    spec = ObjectSpec(
        source="cube",
        object_id="cube",
        cube_half_size=0.02,
        cube_color=(1.0, 0.0, 0.0, 1.0),
    )
    geometry = resolve_object_geometry(spec)
    candidates = AntipodalGraspProvider(_small_config()).generate(geometry, seed=9)
    cache = GraspCache(tmp_path)
    key = "a" * 64

    cache.store(key, candidates, manifest={"seed": 9})
    restored, manifest = cache.load(key)

    assert [item.candidate_id for item in restored] == [
        item.candidate_id for item in candidates
    ]
    assert manifest["candidate_count"] == len(candidates)

    negative_key = "b" * 64
    cache.store(
        negative_key,
        [],
        manifest={"seed": 10},
        failure="proposal-empty",
    )
    negative, negative_manifest = cache.load(negative_key)
    assert negative == []
    assert negative_manifest["failure"] == "proposal-empty"


def test_gripper_oracle_reports_recall_curve() -> None:
    transform = np.eye(4)
    geometry = ResolvedObjectGeometry(
        object_spec=ObjectSpec(
            source="cube",
            object_id="cube",
            cube_half_size=0.02,
            cube_color=(1.0, 0.0, 0.0, 1.0),
        ),
        proposal_mesh=trimesh.creation.box(extents=[0.04] * 3),
        collision_meshes=(),
        object_T_mesh=np.eye(4),
        source_files=(),
        collision_source="primitive",
        canonical_geometry_hash="0" * 64,
    )
    candidates = AntipodalGraspProvider(_small_config(maximum_candidates=4)).generate(
        geometry, seed=2
    )

    result = evaluate_gripper_oracle(
        candidates,
        lambda candidate: GraspOracleAttempt(
            candidate_id=candidate.candidate_id,
            success=candidate.candidate_id == candidates[2].candidate_id,
            reason=(
                "accepted"
                if candidate.candidate_id == candidates[2].candidate_id
                else "lift"
            ),
            max_lift_height=0.1,
            robust_hold_steps=10,
        ),
        ks=(1, 4),
    )

    assert result.recall_at_k == {1: False, 4: True}
