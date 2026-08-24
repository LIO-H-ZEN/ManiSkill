import hashlib
import os
import pathlib
import tempfile
import xml.etree.ElementTree as ET

import mplib
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.two_finger_gripper.motionplanner import (
    TwoFingerGripperMotionPlanningSolver,
)


class PiperMotionPlanningSolver(TwoFingerGripperMotionPlanningSolver):
    """Single-environment PIPER solver that stops at episode boundaries."""

    OPEN = 1.0
    CLOSED = -1.0
    MOVE_GROUP = "piper_tcp"

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = False,
        base_pose: sapien.Pose | None = None,
        visualize_target_grasp_pose: bool = False,
        print_env_info: bool = False,
        joint_vel_limits: float = 0.9,
        joint_acc_limits: float = 0.9,
    ):
        if env.unwrapped.num_envs != 1:
            raise ValueError("PIPER MPLib expert requires num_envs=1")
        if env.unwrapped.control_mode != "pd_joint_pos":
            raise ValueError("PIPER MPLib expert requires pd_joint_pos")
        super().__init__(
            env,
            debug,
            vis,
            base_pose,
            visualize_target_grasp_pose,
            print_env_info,
            joint_vel_limits,
            joint_acc_limits,
        )
        self.episode_done = False
        self.last_transition = None

    @staticmethod
    def _build_kinematic_planning_urdf(source_path: str) -> pathlib.Path:
        source = pathlib.Path(source_path)
        source_bytes = source.read_bytes()
        digest = hashlib.sha256(source_bytes).hexdigest()
        output_dir = pathlib.Path(tempfile.gettempdir()) / "maniskill_piper_mplib"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"piper_kinematic_{digest}.urdf"
        if output_path.exists():
            return output_path

        root = ET.fromstring(source_bytes)
        for link in root.findall("link"):
            for tag in ("collision", "visual", "inertial"):
                for node in list(link.findall(tag)):
                    link.remove(node)
        contents = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        temporary_path = output_path.with_suffix(f".{os.getpid()}.tmp")
        temporary_path.write_bytes(contents)
        os.replace(temporary_path, output_path)
        return output_path

    def setup_planner(self):
        planning_urdf = self._build_kinematic_planning_urdf(
            str(self.env_agent.urdf_path)
        )
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=str(planning_urdf),
            srdf=str(self.env_agent.urdf_path).replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=self.MOVE_GROUP,
        )
        planner.joint_vel_limits = (
            np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
        )
        planner.joint_acc_limits = (
            np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
        )
        return planner

    def _transform_pose_for_planning(self, target: sapien.Pose) -> sapien.Pose:
        # MPLib 0.1.1 fails to convert world targets for a non-zero base pose.
        # Keep its planning base at identity and provide explicit base-frame poses.
        return self.base_pose.inv() * target

    @staticmethod
    def _as_bool(value) -> bool:
        return bool(np.asarray(value).reshape(-1)[0])

    def _step(self, action: np.ndarray):
        if self.episode_done:
            raise RuntimeError("Attempted to step a finished episode")
        transition = self.env.step(np.asarray(action, dtype=np.float32))
        self.last_transition = transition
        _, reward, terminated, truncated, info = transition
        self.elapsed_steps += 1
        self.episode_done = self._as_bool(terminated) or self._as_bool(truncated)
        if self.print_env_info:
            print(f"[{self.elapsed_steps:3}] reward={reward} info={info}")
        if self.vis:
            self.base_env.render_human()
        return transition

    def follow_path(self, result, refine_steps: int = 0):
        positions = np.asarray(result["position"])
        if positions.ndim != 2 or positions.shape[1] != 6 or len(positions) == 0:
            raise ValueError(f"Invalid PIPER path shape: {positions.shape}")
        transition = self.last_transition
        for index in range(len(positions) + refine_steps):
            qpos = positions[min(index, len(positions) - 1)]
            transition = self._step(np.hstack([qpos, self.gripper_state]))
            if self.episode_done:
                break
        return transition

    def _set_gripper(self, state: float, steps: int):
        if not -1.0 <= state <= 1.0:
            raise ValueError(f"Invalid normalized gripper command: {state}")
        if steps <= 0:
            raise ValueError(f"Gripper steps must be positive, got {steps}")
        self.gripper_state = state
        transition = self.last_transition
        for _ in range(steps):
            qpos = self.robot.get_qpos()[0, :6].cpu().numpy()
            transition = self._step(np.hstack([qpos, self.gripper_state]))
            if self.episode_done:
                break
        return transition

    def open_gripper(self, t: int = 6, gripper_state: float | None = None):
        return self._set_gripper(
            self.OPEN if gripper_state is None else gripper_state, t
        )

    def close_gripper(self, t: int = 6, gripper_state: float | None = None):
        return self._set_gripper(
            self.CLOSED if gripper_state is None else gripper_state, t
        )

    def move_to_joint_target(self, target_qpos: np.ndarray, steps: int = 20):
        target_qpos = np.asarray(target_qpos, dtype=np.float32)
        if target_qpos.shape != (6,) or not np.all(np.isfinite(target_qpos)):
            raise ValueError(f"Invalid PIPER arm target: {target_qpos}")
        if steps <= 0:
            raise ValueError(f"Joint interpolation steps must be positive, got {steps}")
        start_qpos = self.robot.get_qpos()[0, :6].cpu().numpy()
        transition = self.last_transition
        for target in np.linspace(start_qpos, target_qpos, steps + 1)[1:]:
            transition = self._step(np.hstack([target, self.gripper_state]))
            if self.episode_done:
                break
        return transition
