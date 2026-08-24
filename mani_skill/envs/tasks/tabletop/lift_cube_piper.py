from typing import Any

import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots.piper import Piper, PiperWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from mani_skill.agents.robots.piper.piper_wristcam import PIPER_CAMERA_INTRINSIC


PIPER_UIDS = frozenset({"piper", "piper_wristcam"})
LIFT_HEIGHT = 0.10
SUCCESS_STREAK_STEPS = 3


def update_success_streak(
    success_streak: torch.Tensor,
    streak_updated_at: torch.Tensor,
    elapsed_steps: torch.Tensor,
    instantaneous_success: torch.Tensor,
) -> None:
    """Update each environment at most once for a distinct control step."""
    new_control_step = elapsed_steps != streak_updated_at
    update_mask = new_control_step & instantaneous_success
    clear_mask = new_control_step & ~instantaneous_success
    success_streak[update_mask] += 1
    success_streak[clear_mask] = 0
    streak_updated_at[new_control_step] = elapsed_steps[new_control_step]


@register_env("LiftCubePiper-v1", max_episode_steps=100)
class LiftCubePiperEnv(BaseEnv):
    """Grasp the red cube and lift it 10 cm for three control steps."""

    SUPPORTED_ROBOTS = ["piper", "piper_wristcam"]
    agent: Piper | PiperWristCam

    def __init__(
        self,
        *args,
        robot_uids: str = "piper_wristcam",
        robot_init_qpos_noise: float = 0.02,
        **kwargs,
    ):
        if robot_uids not in PIPER_UIDS:
            raise ValueError(
                f"LiftCubePiper-v1 requires one of {sorted(PIPER_UIDS)}, got {robot_uids!r}"
            )
        self.robot_init_qpos_noise = robot_init_qpos_noise
        cfg = PICK_CUBE_CONFIGS["piper"]
        self.cube_half_size = cfg["cube_half_size"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        base_pose = sapien_utils.look_at(
            eye=[-0.10, 0.0, 0.45],
            target=[0.03, 0.0, 0.08],
            up=[0.0, 0.0, 1.0],
        )
        side_pose = sapien_utils.look_at(
            eye=[0.18, -0.30, 0.26],
            target=[0.03, 0.0, 0.08],
            up=[0.0, 0.0, 1.0],
        )
        return [
            CameraConfig(
                uid="base_camera",
                pose=base_pose,
                width=224,
                height=224,
                fov=None,
                near=0.01,
                far=100,
                intrinsic=PIPER_CAMERA_INTRINSIC.copy(),
            ),
            CameraConfig(
                uid="side_camera",
                pose=side_pose,
                width=224,
                height=224,
                fov=None,
                near=0.01,
                far=100,
                intrinsic=PIPER_CAMERA_INTRINSIC.copy(),
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.3, 0.5, 0.40],
            target=[0.03, 0.0, 0.08],
            up=[0.0, 0.0, 1.0],
        )
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.35, 0.0, 0.0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.cube_rest_z = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.success_streak = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.streak_updated_at = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            batch_size = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((batch_size, 3))
            xyz[:, :2] = (
                torch.rand((batch_size, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = self.cube_half_size
            quaternions = randomization.random_quaternions(
                batch_size, lock_x=True, lock_y=True
            )
            self.cube.set_pose(Pose.create_from_pq(xyz, quaternions))
            self.cube_rest_z[env_idx] = xyz[:, 2]
            self.success_streak[env_idx] = 0
            self.streak_updated_at[env_idx] = 0

    def _get_obs_extra(self, info: dict):
        return {}

    def evaluate(self):
        lift_height = self.cube.pose.p[:, 2] - self.cube_rest_z
        is_lifted = lift_height >= LIFT_HEIGHT
        is_grasped = self.agent.is_grasping(self.cube)
        instantaneous_success = is_lifted & is_grasped
        update_success_streak(
            self.success_streak,
            self.streak_updated_at,
            self._elapsed_steps,
            instantaneous_success,
        )
        return {
            "success": self.success_streak >= SUCCESS_STREAK_STEPS,
            "is_lifted_10cm": is_lifted,
            "is_grasped": is_grasped,
            "lift_height": lift_height,
            "success_streak": self.success_streak.clone(),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        is_grasped = info["is_grasped"]
        reward += is_grasped
        lift_progress = torch.clamp(info["lift_height"] / LIFT_HEIGHT, 0.0, 1.0)
        reward += 2.0 * lift_progress * is_grasped
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0
