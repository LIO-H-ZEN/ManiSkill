import hashlib
import json
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
from mani_skill.examples.motionplanning.piper.grasping.contracts import (
    GraspCandidate,
    GraspProviderName,
)
from mani_skill.examples.motionplanning.piper.grasping.geometry import (
    ResolvedObjectGeometry,
    _load_mesh,
    resolve_object_geometry,
)
from mani_skill.examples.motionplanning.piper.grasping.oracle import (
    GraspOracleAttempt,
    evaluate_gripper_oracle,
)
from mani_skill.examples.motionplanning.piper.solutions.lift_anything import (
    _limit_candidate_pool_diversely,
)


def _small_config(**overrides) -> AntipodalConfig:
    values = {
        "area_samples": 192,
        "balanced_samples": 96,
        "maximum_opposites_per_point": 3,
        "maximum_contact_pairs": 64,
        "approaches_per_pair": 4,
        "maximum_candidates": 24,
        "maximum_per_contact_region": 8,
    }
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


def test_geometry_loader_removes_degenerate_faces(tmp_path: pathlib.Path) -> None:
    mesh_path = tmp_path / "degenerate.obj"
    mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 2 0 0\nf 1 2 3\nf 1 2 4\n")

    mesh = _load_mesh(mesh_path, 1.0)

    assert len(mesh.faces) == 1
    assert np.all(mesh.area_faces > 0.0)


def test_robodojo_v4_geometry_preserves_disconnected_collision_hulls(
    tmp_path: pathlib.Path,
) -> None:
    asset_root = tmp_path / "assets"
    asset_dir = asset_root / "Rigid" / "test" / "00000"
    asset_dir.mkdir(parents=True)
    visual_path = asset_dir / "visual.glb"
    trimesh.Scene(trimesh.creation.box(extents=[0.08, 0.03, 0.02])).export(visual_path)
    left = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
    left.apply_translation([-0.03, 0.0, 0.0])
    right = left.copy()
    right.apply_translation([0.06, 0.0, 0.0])
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
        "visual.glb_sha256": hashlib.sha256(visual_path.read_bytes()).hexdigest(),
        "collision.ply_sha256": hashlib.sha256(collision_path.read_bytes()).hexdigest(),
    }
    canonical = lambda payload: (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    (asset_dir / "conversion.json").write_bytes(canonical(conversion))
    manifest = {"assets": [conversion]}
    manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    (asset_root / "manifest.json").write_bytes(canonical(manifest))

    geometry = resolve_object_geometry(
        ObjectSpec(source="robodojo", category="test", object_id="00000"),
        asset_root=asset_root,
    )

    assert len(geometry.collision_meshes) == 2
    assert geometry.collision_source == "mesh"
    assert all(
        mesh.is_watertight and mesh.is_volume for mesh in geometry.collision_meshes
    )


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


def test_antipodal_provider_selects_distinct_pairs_before_extra_rolls() -> None:
    spec = ObjectSpec(
        source="cube",
        object_id="cube",
        cube_half_size=0.02,
        cube_color=(1.0, 0.0, 0.0, 1.0),
    )
    geometry = resolve_object_geometry(spec)
    provider = AntipodalGraspProvider(
        _small_config(
            contact_region_size=1.0,
            maximum_candidates=8,
            maximum_per_contact_region=8,
            maximum_per_contact_pair=2,
        )
    )

    candidates = provider.generate(geometry, seed=123)

    assert len(candidates) == 8
    first_pair_indices = [item.metadata["pair_index"] for item in candidates[:4]]
    assert len(set(first_pair_indices)) == 4
    assert all("midpoint_to_com" in item.metadata for item in candidates)


def test_antipodal_provider_samples_the_physics_collision_surface() -> None:
    collision = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
    geometry = ResolvedObjectGeometry(
        object_spec=ObjectSpec(
            source="cube",
            object_id="cube",
            cube_half_size=0.05,
            cube_color=(1.0, 0.0, 0.0, 1.0),
        ),
        proposal_mesh=trimesh.creation.box(extents=[0.1, 0.1, 0.1]),
        collision_meshes=(collision,),
        object_T_mesh=np.eye(4),
        source_files=(),
        collision_source="mesh",
        canonical_geometry_hash="0" * 64,
    )

    candidates = AntipodalGraspProvider(_small_config(maximum_candidates=8)).generate(
        geometry, seed=4
    )

    contacts = np.concatenate([item.contact_points for item in candidates], axis=0)
    assert np.abs(contacts).max() <= 0.010001
    assert all(item.metadata["contact_surface"] == "mesh" for item in candidates)


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
    mesh = trimesh.creation.box(extents=[0.04] * 3)
    geometry = ResolvedObjectGeometry(
        object_spec=ObjectSpec(
            source="cube",
            object_id="cube",
            cube_half_size=0.02,
            cube_color=(1.0, 0.0, 0.0, 1.0),
        ),
        proposal_mesh=mesh,
        collision_meshes=(mesh.copy(),),
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


def test_smoke_candidate_limit_preserves_approach_diversity() -> None:
    candidates = []
    for index, approach in enumerate(
        ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    ):
        for duplicate in range(3):
            transform = np.eye(4)
            transform[:3, 2] = approach
            # Complete a right-handed basis for the synthetic candidate.
            transform[:3, 1] = [0.0, 1.0, 0.0] if index != 1 else [1.0, 0.0, 0.0]
            transform[:3, 0] = np.cross(transform[:3, 1], transform[:3, 2])
            candidates.append(
                GraspCandidate(
                    candidate_id=f"candidate-{index}-{duplicate}",
                    source=GraspProviderName.ANTIPODAL,
                    object_T_tcp=transform,
                    required_width=0.02,
                    proposal_score=1.0 - duplicate * 0.01,
                    contact_points=np.zeros((2, 3)),
                    metadata={"contact_region": (0, 0, 0)},
                )
            )

    selected = _limit_candidate_pool_diversely(candidates, 3)

    assert {
        tuple(np.round(item.object_T_tcp[:3, 2]).astype(int)) for item in selected
    } == {(0, 0, 1), (0, 1, 0), (1, 0, 0)}


def test_smoke_candidate_limit_selects_distinct_pairs_before_extra_rolls() -> None:
    candidates = []
    for pair_index in range(2):
        for roll_index, approach in enumerate(([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])):
            transform = np.eye(4)
            transform[:3, 2] = approach
            transform[:3, 1] = [0.0, 1.0, 0.0]
            if roll_index == 1:
                transform[:3, 1] = [0.0, 0.0, 1.0]
            transform[:3, 0] = np.cross(transform[:3, 1], transform[:3, 2])
            candidates.append(
                GraspCandidate(
                    candidate_id=f"pair-{pair_index}-roll-{roll_index}",
                    source=GraspProviderName.ANTIPODAL,
                    object_T_tcp=transform,
                    required_width=0.02,
                    proposal_score=1.0 - pair_index * 0.01,
                    contact_points=np.array(
                        [
                            [-0.01, pair_index * 0.01, 0.0],
                            [0.01, pair_index * 0.01, 0.0],
                        ]
                    ),
                    metadata={
                        "contact_region": (0, 0, 0),
                        "pair_index": pair_index,
                        "roll_index": roll_index,
                        "midpoint_to_com": pair_index * 0.01,
                    },
                )
            )

    selected = _limit_candidate_pool_diversely(candidates, 2)

    assert [item.metadata["pair_index"] for item in selected] == [0, 1]
