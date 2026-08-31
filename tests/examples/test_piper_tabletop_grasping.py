from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from mani_skill.envs.tasks.pick_anything.episode_specs import ObjectSpec
from mani_skill.examples.motionplanning.piper.grasping.contracts import (
    GraspProviderName,
)
from mani_skill.examples.motionplanning.piper.grasping.geometry import (
    ResolvedObjectGeometry,
)
from mani_skill.examples.motionplanning.piper.grasping.tabletop import (
    TABLETOP_OFFSET_LATERAL_FRACTIONS,
    TabletopConfig,
    TabletopGraspProvider,
    _line_hits,
    _union_line_hits,
)
from mani_skill.examples.motionplanning.piper.solutions import lift_anything


def _box_geometry(extents=(0.04, 0.06, 0.03)) -> ResolvedObjectGeometry:
    mesh = trimesh.creation.box(extents=extents)
    return ResolvedObjectGeometry(
        object_spec=ObjectSpec(
            source="cube",
            object_id="cube",
            cube_half_size=0.02,
            cube_color=(1.0, 0.0, 0.0, 1.0),
        ),
        proposal_mesh=mesh.copy(),
        collision_meshes=(mesh,),
        object_T_mesh=np.eye(4),
        source_files=(),
        collision_source="primitive",
        canonical_geometry_hash="0" * 64,
    )


def test_line_hits_return_two_box_surfaces() -> None:
    mesh = trimesh.creation.box(extents=[0.04, 0.06, 0.03])

    hits = _line_hits(
        mesh,
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        merge_tolerance=1e-5,
    )

    assert len(hits) == 2
    assert hits[1].distance - hits[0].distance == pytest.approx(0.04)
    np.testing.assert_allclose(hits[0].normal, [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(hits[1].normal, [1.0, 0.0, 0.0])


def test_union_line_hits_removes_internal_convex_component_surfaces() -> None:
    left = trimesh.creation.box(extents=[0.03, 0.04, 0.03])
    left.apply_translation([-0.01, 0.0, 0.0])
    right = trimesh.creation.box(extents=[0.03, 0.04, 0.03])
    right.apply_translation([0.01, 0.0, 0.0])

    hits = _union_line_hits(
        (left, right),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        merge_tolerance=1e-5,
    )

    assert len(hits) == 2
    assert hits[1].distance - hits[0].distance == pytest.approx(0.05)
    np.testing.assert_allclose(hits[0].normal, [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(hits[1].normal, [1.0, 0.0, 0.0])


def test_tabletop_provider_generates_deterministic_top_down_pinches() -> None:
    geometry = _box_geometry()
    config = TabletopConfig(
        yaw_samples=6,
        height_fractions=(0.5,),
        lateral_fractions=(0.0,),
        approach_tilts_degrees=(0.0,),
        maximum_candidates=12,
    )
    provider = TabletopGraspProvider(config)
    kwargs = {
        "world_T_object": np.eye(4),
        "robot_base_position": np.array([-0.3, 0.0, 0.0]),
    }

    first = provider.generate(geometry, **kwargs)
    second = provider.generate(geometry, **kwargs)

    assert first
    assert [item.candidate_id for item in first] == [
        item.candidate_id for item in second
    ]
    assert all(item.source is GraspProviderName.TABLETOP for item in first)
    assert all(item.object_T_tcp[2, 2] == pytest.approx(-1.0) for item in first)
    assert all(0.001 <= item.required_width <= 0.068 for item in first)
    for left, right in zip(first, second):
        np.testing.assert_allclose(left.object_T_tcp, right.object_T_tcp)


def test_tabletop_provider_respects_world_up_for_rotated_objects() -> None:
    geometry = _box_geometry()
    angle = np.deg2rad(30.0)
    world_T_object = np.eye(4)
    world_T_object[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    provider = TabletopGraspProvider(
        TabletopConfig(
            yaw_samples=4,
            height_fractions=(0.5,),
            lateral_fractions=(0.0,),
            approach_tilts_degrees=(0.0,),
            maximum_candidates=8,
        )
    )

    candidates = provider.generate(
        geometry,
        world_T_object=world_T_object,
        robot_base_position=np.array([-0.3, 0.0, 0.0]),
    )

    for candidate in candidates:
        world_approach = world_T_object[:3, :3] @ candidate.object_T_tcp[:3, 2]
        np.testing.assert_allclose(world_approach, [0.0, 0.0, -1.0], atol=1e-7)


def test_tabletop_provider_keeps_small_approach_tilt_variants() -> None:
    provider = TabletopGraspProvider(
        TabletopConfig(
            yaw_samples=2,
            height_fractions=(0.5,),
            lateral_fractions=(0.0,),
            approach_tilts_degrees=(0.0, 7.5),
            approach_azimuth_offsets_degrees=(20.0,),
            maximum_candidates=8,
        )
    )

    candidates = provider.generate(
        _box_geometry(),
        world_T_object=np.eye(4),
        robot_base_position=np.array([-0.3, 0.0, 0.0]),
    )

    tilts_by_pair: dict[int, set[float]] = {}
    for candidate in candidates:
        pair_index = int(candidate.metadata["pair_index"])
        tilts_by_pair.setdefault(pair_index, set()).add(
            float(candidate.metadata["tilt_degrees"])
        )
    assert any(tilts == {0.0, 7.5} for tilts in tilts_by_pair.values())


def test_default_tabletop_grid_includes_fine_tabletop_offsets() -> None:
    config = TabletopConfig()

    assert config.yaw_samples == 24
    assert 0.45 in config.height_fractions
    assert config.lateral_fractions == (0.0, -0.15, 0.15)
    assert TABLETOP_OFFSET_LATERAL_FRACTIONS == (-0.3, 0.3, -0.45, 0.45)
    assert config.approach_tilts_degrees[0] == 7.5


def test_offset_tabletop_grid_reaches_off_center_regions_of_long_objects() -> None:
    candidates = TabletopGraspProvider(
        TabletopConfig(lateral_fractions=TABLETOP_OFFSET_LATERAL_FRACTIONS)
    ).generate(
        _box_geometry(extents=(0.26, 0.04, 0.02)),
        world_T_object=np.eye(4),
        robot_base_position=np.array([-0.3, 0.0, 0.0]),
    )

    midpoints = np.asarray([item.object_T_tcp[:3, 3] for item in candidates])
    assert np.max(np.abs(midpoints[:, 0])) >= 0.1


def test_common_solver_reports_empty_tabletop_proposals(monkeypatch) -> None:
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(num_envs=1, success_streak_steps=10),
        reset=lambda **kwargs: None,
    )
    monkeypatch.setattr(
        lift_anything,
        "_tabletop_candidates",
        lambda base_env: (_ for _ in ()).throw(
            RuntimeError("proposal-empty: no valid tabletop pinch")
        ),
    )

    result = lift_anything.solve(
        env,
        seed=0,
        provider=GraspProviderName.TABLETOP,
        pipeline="common",
    )

    assert not result.success
    assert result.reason == "proposal-empty: no valid tabletop pinch"
    assert result.attempted_candidates == 0
    assert result.evaluations == ()


def test_tabletop_provider_fast_fails_when_no_width_is_executable() -> None:
    geometry = _box_geometry(extents=(0.09, 0.09, 0.03))
    provider = TabletopGraspProvider(
        TabletopConfig(
            yaw_samples=4,
            height_fractions=(0.5,),
            lateral_fractions=(0.0,),
            approach_tilts_degrees=(0.0,),
        )
    )

    with pytest.raises(RuntimeError, match="proposal-empty"):
        provider.generate(
            geometry,
            world_T_object=np.eye(4),
            robot_base_position=np.array([-0.3, 0.0, 0.0]),
        )
