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
from mani_skill.examples.motionplanning.piper.motionplanner import (
    PiperMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.piper.solutions.lift_anything import (
    _path_end_robot_qpos,
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


def test_path_end_robot_qpos_preserves_arm_and_sets_gripper_width() -> None:
    result = {
        "position": np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]
        )
    }
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.035, -0.035])

    qpos = _path_end_robot_qpos(result, start, gripper_width=0.04)

    np.testing.assert_allclose(qpos[:6], result["position"][-1])
    np.testing.assert_allclose(qpos[6:], [0.02, -0.02])


def test_explicit_qpos_screw_planning_retries_once() -> None:
    calls = []

    class FakePlanner:
        def plan_screw(self, payload, start_qpos, **kwargs):
            calls.append((payload.copy(), start_qpos.copy(), kwargs))
            return {
                "status": "Success" if len(calls) == 2 else "screw plan failed",
                "position": np.zeros((1, 6)),
            }

    solver = object.__new__(PiperMotionPlanningSolver)
    solver.robot = SimpleNamespace(get_qpos=lambda: np.zeros((1, 8)))
    solver.base_pose = sapien.Pose()
    solver.base_env = SimpleNamespace(control_timestep=0.05)
    solver.use_point_cloud = True
    solver.planner = FakePlanner()
    start_qpos = np.arange(8, dtype=np.float64)

    result = solver.plan_screw_from_qpos(sapien.Pose([0.1, 0.2, 0.3]), start_qpos)

    assert result["status"] == "Success"
    assert len(calls) == 2
    np.testing.assert_allclose(calls[0][1], start_qpos)
    assert calls[0][2] == {"time_step": 0.05, "use_point_cloud": True}


def test_gripper_clearance_checks_the_full_closure_sweep() -> None:
    geometry = PiperGripperGeometry.from_package_assets()
    transform = np.eye(4)
    transform[2, 3] = 0.20

    clearance = minimum_gripper_clearance(transform, geometry, contact_width=0.01)

    assert np.isfinite(clearance)
    assert clearance > 0.0


def test_gripper_pad_includes_the_inner_fingertip_surface() -> None:
    geometry = PiperGripperGeometry.from_package_assets()
    bounds = geometry.finger_bounds["link7"]
    fingertip_pad = np.array([0.0, -0.005, bounds[0, 2]])
    outer_surface = np.array([0.0, -0.005, bounds[1, 2]])

    assert geometry.is_pad_point("link7", np.eye(4), fingertip_pad)
    assert not geometry.is_pad_point("link7", np.eye(4), outer_surface)


def test_gripper_pad_contact_projects_fcl_point_to_reported_face() -> None:
    geometry = PiperGripperGeometry.from_package_assets()
    vertices = geometry.finger_vertices["link7"]
    faces = geometry.finger_faces["link7"]
    triangles = vertices[faces]
    minimum_z = geometry.finger_bounds["link7"][0, 2]
    pad_faces = np.flatnonzero(
        np.max(np.abs(triangles[:, :, 2] - minimum_z), axis=1) < 1e-8
    )
    non_pad_faces = np.flatnonzero(triangles[:, :, 2].min(axis=1) > minimum_z + 0.005)
    assert len(pad_faces) > 0
    assert len(non_pad_faces) > 0

    # FCL may report a representative point well outside the finger surface.
    # The triangle index remains reliable, so classification must project the
    # point onto that triangle before checking the pad region.
    off_surface_point = np.array([-0.05, 0.0, 0.01])
    assert geometry.is_pad_contact(
        "link7", np.eye(4), off_surface_point, int(pad_faces[0])
    )
    assert not geometry.is_pad_contact(
        "link7", np.eye(4), off_surface_point, int(non_pad_faces[0])
    )

    with pytest.raises(ValueError, match="Invalid link7 collision face index"):
        geometry.is_pad_contact("link7", np.eye(4), off_surface_point, -1)


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


def test_target_geometry_validator_reuses_gripper_bvhs(monkeypatch) -> None:
    pytest.importorskip("fcl")
    geometry = PiperGripperGeometry.from_package_assets()
    target = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
    target.apply_translation([10.0, 10.0, 10.0])
    base_env = SimpleNamespace(obj=SimpleNamespace(pose=sapien.Pose()))
    monkeypatch.setattr(pipeline, "actor_collision_meshes", lambda actor: (target,))
    original_get_fcl_obj = trimesh.collision.CollisionManager._get_fcl_obj
    build_count = 0

    def counted_get_fcl_obj(manager, mesh):
        nonlocal build_count
        build_count += 1
        return original_get_fcl_obj(manager, mesh)

    monkeypatch.setattr(
        trimesh.collision.CollisionManager,
        "_get_fcl_obj",
        counted_get_fcl_obj,
    )
    validator = TargetGeometryCollisionValidator(base_env, geometry)
    initialization_builds = build_count

    for _ in range(2):
        assert validator.validate_pose(
            np.eye(4), width=0.068, allow_pad_contact=False, progress=0.0
        ) == frozenset()

    assert build_count == initialization_builds


def test_target_geometry_validator_samples_descend_and_closure() -> None:
    validator = object.__new__(TargetGeometryCollisionValidator)
    calls = []
    def validate_pose(transform, **kwargs):
        calls.append((transform.copy(), kwargs))
        return (
            frozenset({"link7", "link8"})
            if kwargs["width"] <= 0.02
            else frozenset()
        )

    validator.validate_pose = validate_pose
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
    assert all(call[1]["progress"] == 1.0 for call in closure)


def test_target_geometry_validator_stops_at_first_bilateral_pad_contact() -> None:
    validator = object.__new__(TargetGeometryCollisionValidator)
    widths = []

    def validate_pose(transform, **kwargs):
        del transform
        widths.append(kwargs["width"])
        return (
            frozenset({"link7", "link8"})
            if kwargs["width"] <= 0.04
            else frozenset()
        )

    validator.validate_pose = validate_pose
    width = validator.validate_descend_and_closure(
        sapien.Pose([0.0, 0.0, 0.07]),
        sapien.Pose(),
        contact_width=0.02,
        closure_samples=5,
    )

    assert width == pytest.approx(0.032)
    assert widths[-1] == pytest.approx(0.032)


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
