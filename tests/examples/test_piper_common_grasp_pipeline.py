from types import SimpleNamespace

import numpy as np
import sapien

from mani_skill.examples.motionplanning.piper.grasping.contracts import (
    GraspCandidate,
    GraspProviderName,
)
from mani_skill.examples.motionplanning.piper.grasping.gripper_geometry import (
    PiperGripperGeometry,
)
from mani_skill.examples.motionplanning.piper.grasping.pipeline import (
    RankedCandidate,
    dense_translation_waypoints,
    minimum_gripper_clearance,
    rank_candidates,
    world_grasp_pose,
)


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


def test_rank_candidates_uses_buckets_and_contact_diversity() -> None:
    candidates = [
        RankedCandidate(
            candidate=_candidate(f"c{index}", score=0.91, region=(index // 5, 0, 0)),
            clearance_score=0.02,
            ik_cost=float(index),
            path_length=0.0,
        )
        for index in range(10)
    ]

    selected = rank_candidates(candidates, maximum_candidates=8)

    assert len(selected) == 8
    assert (
        sum(item.candidate.metadata["contact_region"] == (0, 0, 0) for item in selected)
        == 4
    )
