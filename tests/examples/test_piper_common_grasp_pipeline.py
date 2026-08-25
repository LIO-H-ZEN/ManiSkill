from types import SimpleNamespace

import numpy as np
import pytest
import sapien
import trimesh

from mani_skill.examples.motionplanning.piper.grasping.contracts import (
    GraspCandidate,
    GraspProviderName,
)
from mani_skill.examples.motionplanning.piper.grasping.gripper_geometry import (
    PiperGripperGeometry,
)
from mani_skill.examples.motionplanning.piper.grasping.pipeline import (
    RankedCandidate,
    TargetGeometryCollisionValidator,
    dense_translation_waypoints,
    minimum_gripper_clearance,
    rank_candidates,
    world_grasp_pose,
)
from mani_skill.examples.motionplanning.piper.grasping import pipeline


def _candidate(candidate_id: str, *, score: float, region=(0, 0, 0)):
    return GraspCandidate(
        candidate_id=candidate_id,
        source=GraspProviderName.ANTIPODAL,
        object_T_tcp=np.eye(4),
        required_width=0.02,
        proposal_score=score,
        contact_points=np.array([[-0.01, 0, 0], [0.01, 0, 0]]),
        metadata={"contact_region": region},
    )


def test_world_grasp_pose_is_recomposed_from_current_object_pose() -> None:
    candidate = _candidate("candidate", score=1.0)
    candidate.object_T_tcp[:3, 3] = [0.01, 0.0, 0.02]

    first = world_grasp_pose(sapien.Pose([0.1, 0.0, 0.0]), candidate)
    second = world_grasp_pose(sapien.Pose([0.2, 0.0, 0.0]), candidate)

    np.testing.assert_allclose(first.p, [0.11, 0.0, 0.02])
    np.testing.assert_allclose(second.p, [0.21, 0.0, 0.02])


def test_dense_descend_waypoints_respect_two_millimetre_limit() -> None:
    waypoints = dense_translation_waypoints(np.array([0.0, 0.0, 0.07]), np.zeros(3))
    points = np.vstack([[0.0, 0.0, 0.07], waypoints])

    assert len(waypoints) == 35
    assert np.linalg.norm(np.diff(points, axis=0), axis=1).max() <= 0.002 + 1e-12


def test_gripper_clearance_checks_the_full_closure_sweep() -> None:
    geometry = PiperGripperGeometry.from_package_assets()
    transform = np.eye(4)
    transform[2, 3] = 0.20

    clearance = minimum_gripper_clearance(transform, geometry, contact_width=0.01)

    assert np.isfinite(clearance)
    assert clearance > 0.0


def test_target_geometry_validator_rejects_gripper_base_collision(monkeypatch) -> None:
    pytest.importorskip("fcl")
    geometry = PiperGripperGeometry.from_package_assets()
    tcp_T_base = geometry.tcp_link_transforms(0.068)["gripper_base"]
    base_mesh = geometry.link_meshes()["gripper_base"]
    center = (
        tcp_T_base[:3, :3] @ base_mesh.centroid + tcp_T_base[:3, 3]
    )
    target = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
    target.apply_translation(center)
    base_env = SimpleNamespace(obj=SimpleNamespace(pose=sapien.Pose()))
    monkeypatch.setattr(pipeline, "actor_collision_meshes", lambda actor: (target,))
    validator = TargetGeometryCollisionValidator(base_env, geometry)

    with pytest.raises(RuntimeError, match="target-contact-gripper_base"):
        validator.validate_pose(
            np.eye(4), width=0.068, allow_pad_contact=False, progress=0.0
        )


def test_target_geometry_validator_samples_descend_and_closure() -> None:
    validator = object.__new__(TargetGeometryCollisionValidator)
    calls = []
    validator.validate_pose = lambda transform, **kwargs: calls.append(
        (transform.copy(), kwargs)
    )
    pregrasp = sapien.Pose([0.0, 0.0, 0.07])
    grasp = sapien.Pose()

    validator.validate_descend_and_closure(
        pregrasp, grasp, contact_width=0.02, closure_samples=8
    )

    descend = calls[:-8]
    closure = calls[-8:]
    assert len(descend) >= 35
    descend_positions = np.asarray([call[0][:3, 3] for call in descend])
    points = np.vstack([np.asarray(pregrasp.p), descend_positions])
    assert np.linalg.norm(np.diff(points, axis=0), axis=1).max() <= 0.002 + 1e-7
    assert all(call[1]["width"] == 0.068 for call in descend)
    np.testing.assert_allclose(
        [call[1]["width"] for call in closure], np.linspace(0.068, 0.02, 8)
    )


def test_rank_candidates_uses_buckets_and_contact_diversity() -> None:
    candidates = [
        RankedCandidate(
            candidate=_candidate(f"c{index}", score=0.91, region=(index // 5, 0, 0)),
            clearance_score=0.02,
            ik_cost=float(index),
            path_length=0.0,
            com_distance=0.0,
            gravity_torque_risk=0.0,
        )
        for index in range(10)
    ]

    selected = rank_candidates(candidates, maximum_candidates=8)

    assert len(selected) == 8
    assert (
        sum(item.candidate.metadata["contact_region"] == (0, 0, 0) for item in selected)
        == 4
    )


def test_rank_candidates_uses_com_and_gravity_cost_after_quality_buckets() -> None:
    candidates = [
        RankedCandidate(
            candidate=_candidate(candidate_id, score=0.91),
            clearance_score=0.02,
            ik_cost=0.5,
            path_length=0.1,
            com_distance=com_distance,
            gravity_torque_risk=gravity_torque_risk,
        )
        for candidate_id, com_distance, gravity_torque_risk in (
            ("high-com", 0.08, 0.01),
            ("high-torque", 0.01, 0.08),
            ("balanced", 0.01, 0.01),
        )
    ]

    selected = rank_candidates(candidates, maximum_candidates=3)

    assert [item.candidate.candidate_id for item in selected] == [
        "balanced",
        "high-com",
        "high-torque",
    ]
