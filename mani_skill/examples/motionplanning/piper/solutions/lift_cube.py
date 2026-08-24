import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.lift_cube_piper import LiftCubePiperEnv
from mani_skill.examples.motionplanning.piper.motionplanner import (
    PiperMotionPlanningSolver,
)


UNTILTED_PLANAR_REACH = 0.36
FULL_TILT_PLANAR_REACH = 0.42
MAX_APPROACH_TILT = np.deg2rad(30.0)


def build_adaptive_grasp_pose(
    agent,
    cube_center: np.ndarray,
    robot_base_position: np.ndarray,
) -> sapien.Pose:
    """Tilt the tool outward when a vertical grasp exceeds PIPER's low workspace."""
    cube_center = np.asarray(cube_center, dtype=np.float64)
    robot_base_position = np.asarray(robot_base_position, dtype=np.float64)
    planar = cube_center[:2] - robot_base_position[:2]
    planar_distance = np.linalg.norm(planar)
    if planar_distance <= 0.0:
        raise ValueError("Cube center must not coincide with the PIPER base axis")
    radial = planar / planar_distance
    tilt_fraction = np.clip(
        (planar_distance - UNTILTED_PLANAR_REACH)
        / (FULL_TILT_PLANAR_REACH - UNTILTED_PLANAR_REACH),
        0.0,
        1.0,
    )
    tilt = MAX_APPROACH_TILT * tilt_fraction
    approaching = np.array(
        [radial[0] * np.sin(tilt), radial[1] * np.sin(tilt), -np.cos(tilt)]
    )
    closing = np.array([-radial[1], radial[0], 0.0])
    return agent.build_grasp_pose(approaching, closing, cube_center)


def _require_success(result, stage: str) -> None:
    if result == -1:
        raise RuntimeError(f"PIPER motion planning failed during {stage}")


def _transition_flag(transition, key: str) -> bool:
    value = transition[4][key]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def solve(
    env: LiftCubePiperEnv,
    *,
    seed: int | None = None,
    debug: bool = False,
    vis: bool = False,
):
    env.reset(seed=seed)
    base_env = env.unwrapped
    planner = PiperMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=base_env.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    try:
        lift_qpos = base_env.agent.robot.get_qpos()[0, :6].cpu().numpy().copy()
        cube_center = base_env.cube.pose.p[0].cpu().numpy()
        grasp_pose = build_adaptive_grasp_pose(
            base_env.agent,
            cube_center,
            planner.base_pose.p,
        )

        if hasattr(env, "expert_stage"):
            env.expert_stage = "pregrasp"
        _require_success(
            planner.move_to_pose_with_RRTConnect(
                grasp_pose * sapien.Pose([0.0, 0.0, -0.06])
            ),
            "pregrasp",
        )
        if planner.episode_done:
            return planner.last_transition

        if hasattr(env, "expert_stage"):
            env.expert_stage = "descend"
        _require_success(
            planner.move_to_pose_with_screw(grasp_pose),
            "descend",
        )
        if planner.episode_done:
            return planner.last_transition

        if hasattr(env, "expert_stage"):
            env.expert_stage = "grasp"
        planner.close_gripper(t=6)
        if planner.episode_done:
            return planner.last_transition
        if not _transition_flag(planner.last_transition, "is_grasped"):
            raise RuntimeError("PIPER gripper failed to grasp the cube")

        if hasattr(env, "expert_stage"):
            env.expert_stage = "lift"
        planner.move_to_joint_target(lift_qpos, steps=20)
        return planner.last_transition
    finally:
        planner.close()
