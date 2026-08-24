"""Bounded OBB-based expert for LiftAnythingPiper-v1."""

from __future__ import annotations

import dataclasses
from typing import Iterable

import numpy as np
import sapien

from mani_skill.envs.tasks.pick_anything.lift_anything_piper import (
    LiftAnythingPiperEnv,
)
from mani_skill.examples.motionplanning.piper.motionplanner import (
    PiperMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.piper.solutions.lift_cube import (
    FULL_TILT_PLANAR_REACH,
    MAX_APPROACH_TILT,
    UNTILTED_PLANAR_REACH,
)


MAX_GRASP_CANDIDATES = 16
MAX_GRIPPER_WIDTH = 0.07


@dataclasses.dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    pose: sapien.Pose
    required_width: float


@dataclasses.dataclass(frozen=True)
class ExpertResult:
    success: bool
    reason: str
    candidate_id: str | None
    attempted_candidates: int
    transition: tuple | None


def _adaptive_approach(center: np.ndarray, robot_base_position: np.ndarray) -> np.ndarray:
    planar = center[:2] - robot_base_position[:2]
    planar_distance = np.linalg.norm(planar)
    if planar_distance <= 0.0:
        raise ValueError("Object center must not coincide with the PIPER base axis")
    radial = planar / planar_distance
    tilt_fraction = np.clip(
        (planar_distance - UNTILTED_PLANAR_REACH)
        / (FULL_TILT_PLANAR_REACH - UNTILTED_PLANAR_REACH),
        0.0,
        1.0,
    )
    tilt = MAX_APPROACH_TILT * tilt_fraction
    return np.array(
        [radial[0] * np.sin(tilt), radial[1] * np.sin(tilt), -np.cos(tilt)]
    )


def _orthogonal_closing(closing: np.ndarray, approach: np.ndarray) -> np.ndarray:
    projected = closing - approach * float(closing @ approach)
    norm = np.linalg.norm(projected)
    if norm < 1e-8:
        raise ValueError("Closing direction is parallel to approach direction")
    return projected / norm


def generate_grasp_candidates(
    agent,
    obj,
    robot_base_position: np.ndarray,
    *,
    max_candidates: int = MAX_GRASP_CANDIDATES,
) -> list[GraspCandidate]:
    if not 1 <= max_candidates <= MAX_GRASP_CANDIDATES:
        raise ValueError(
            f"max_candidates must be in [1, {MAX_GRASP_CANDIDATES}], got {max_candidates}"
        )
    mesh = obj.get_first_collision_mesh(to_world_frame=False)
    if mesh is None:
        raise RuntimeError("invalid-mesh: object has no collision mesh")
    bounds = np.asarray(mesh.bounding_box.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise RuntimeError("invalid-mesh: collision mesh has invalid bounds")
    extents = bounds[1] - bounds[0]
    if np.any(extents <= 0.0):
        raise RuntimeError(f"invalid-mesh: non-positive OBB extents {extents}")

    actor_matrix = np.asarray(obj.pose.to_transformation_matrix()[0].cpu())
    local_center = (bounds[0] + bounds[1]) / 2.0
    center = actor_matrix[:3, :3] @ local_center + actor_matrix[:3, 3]
    axes = [actor_matrix[:3, 0], actor_matrix[:3, 1]]
    approach = _adaptive_approach(center, np.asarray(robot_base_position))
    z_offsets = (0.0, 0.18, -0.18)
    yaw_offsets = np.deg2rad((0.0, 12.0, -12.0))
    candidates: list[GraspCandidate] = []
    feasible_axes = sorted(
        ((float(extents[index]), index) for index in range(2)), key=lambda item: item[0]
    )
    for required_width, axis_index in feasible_axes:
        if required_width > MAX_GRIPPER_WIDTH:
            continue
        base_axis = axes[axis_index]
        for z_fraction in z_offsets:
            grasp_center = center.copy()
            grasp_center[2] += z_fraction * extents[2]
            for yaw in yaw_offsets:
                cosine, sine = np.cos(yaw), np.sin(yaw)
                rotated = np.array(
                    [
                        cosine * base_axis[0] - sine * base_axis[1],
                        sine * base_axis[0] + cosine * base_axis[1],
                        base_axis[2],
                    ]
                )
                closing = _orthogonal_closing(rotated, approach)
                candidate_id = (
                    f"axis{axis_index}-z{z_fraction:+.2f}-yaw{np.rad2deg(yaw):+.0f}"
                )
                candidates.append(
                    GraspCandidate(
                        candidate_id,
                        agent.build_grasp_pose(approach, closing, grasp_center),
                        required_width,
                    )
                )
                if len(candidates) == max_candidates:
                    return candidates
    if not candidates:
        raise RuntimeError(
            f"width-infeasible: horizontal OBB widths {extents[:2].tolist()} exceed "
            f"{MAX_GRIPPER_WIDTH} m"
        )
    return candidates


def _scalar_bool(value) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def _run_candidate(
    env,
    candidate: GraspCandidate,
    *,
    debug: bool,
    vis: bool,
):
    base_env: LiftAnythingPiperEnv = env.unwrapped
    planner = PiperMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=base_env.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    lift_qpos = base_env.agent.robot.get_qpos()[0, :6].cpu().numpy().copy()
    try:
        env.expert_stage = "pregrasp"
        result = planner.move_to_pose_with_RRTConnect(
            candidate.pose * sapien.Pose([0.0, 0.0, -0.07])
        )
        if result == -1:
            raise RuntimeError("pregrasp")
        if planner.episode_done:
            return planner.last_transition

        env.expert_stage = "descend"
        result = planner.move_to_pose_with_screw(candidate.pose)
        if result == -1:
            raise RuntimeError("descend")
        if planner.episode_done:
            return planner.last_transition

        env.expert_stage = "grasp"
        planner.close_gripper(t=6)
        if planner.episode_done:
            return planner.last_transition
        if not _scalar_bool(planner.last_transition[4]["is_grasped"]):
            raise RuntimeError("grasp")

        env.expert_stage = "lift"
        planner.move_to_joint_target(lift_qpos, steps=20)
        return planner.last_transition
    finally:
        planner.close()


def solve(
    env,
    *,
    seed: int,
    candidates: Iterable[GraspCandidate] | None = None,
    debug: bool = False,
    vis: bool = False,
) -> ExpertResult:
    base_env: LiftAnythingPiperEnv = env.unwrapped
    if base_env.num_envs != 1:
        raise ValueError("LiftAnything expert requires num_envs=1")
    env.reset(seed=seed, options={"reconfigure": True})
    if candidates is None:
        candidates = generate_grasp_candidates(
            base_env.agent,
            base_env.obj,
            np.asarray(base_env.agent.robot.pose.p[0].cpu()),
        )
    candidates = list(candidates)
    if not candidates:
        raise ValueError("At least one grasp candidate is required")
    if len(candidates) > MAX_GRASP_CANDIDATES:
        raise ValueError(f"At most {MAX_GRASP_CANDIDATES} candidates may be tested")

    last_reason = "no_transition"
    for index, candidate in enumerate(candidates, start=1):
        env.reset(seed=seed, options={"reconfigure": True})
        try:
            transition = _run_candidate(env, candidate, debug=debug, vis=vis)
        except RuntimeError as error:
            last_reason = str(error)
            continue
        if transition is not None and _scalar_bool(transition[4]["success"]):
            return ExpertResult(True, "accepted", candidate.candidate_id, index, transition)
        last_reason = "timeout" if transition is not None and _scalar_bool(transition[3]) else "lift"
    return ExpertResult(False, last_reason, None, len(candidates), None)
