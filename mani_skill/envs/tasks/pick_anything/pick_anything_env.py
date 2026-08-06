"""PickAnything: a pick task with composable domain randomization.

The task itself is the same as PickCube / PickSingleYCB (grasp an object and
move it to a goal position), but every diversity axis --- object, table,
lighting --- is delegated to a pluggable :class:`Randomizer`.

**Defaults (v1):**
- table: the fixed wood PickCube table (``TableSceneBuilder``)
- object: a composite source randomizer mixing **cube** (procedural), **YCB**
  (cached dataset), and **InternDataAssets** (downloaded meshes); one source is
  drawn per env per reconfiguration. InternDataAssets is a gated HF dataset
  (needs ``huggingface-cli login`` + license acceptance); pass
  ``object_sources=["cube","ycb"]`` for a no-download default.
- lighting: HDRI environment map + ambient/directional light (HDRI is auto
  disabled on macOS due to a MoltenVK bug; see ``HDRILightingRandomizer``)

Swap any axis by passing a custom ``object_randomizer`` / ``table_randomizer``
/ ``lighting_randomizer``, or grow the object candidate set via
``object_sources`` (e.g. ``["cube", "ycb", "interndata"]`` --- InternDataAssets
meshes are downloaded on demand from a gated HF dataset).

**Randomizations:**
- object identity (cube / YCB / mesh) + geometry/color --- per reconfiguration
- table model/material --- per reconfiguration (wood by default)
- HDRI environment map --- per episode (cheap, render-time; off on macOS)
- ambient + directional light direction/intensity --- per reconfiguration
- object xy position + yaw, goal position --- per episode
- robot init qpos noise --- per episode

**Success Conditions:**
- the object position is within ``goal_thresh`` (default 0.025m) of the goal
- the robot is static (q velocity < 0.2)
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .randomization import (
    HDRILightingRandomizer,
    ObjectSource,
    Randomizer,
)
from .randomization.clutter_randomizer import ClutterRandomizer
from .randomization.object_randomizer import CompositeObjectRandomizer
from .randomization.table_randomizer import (
    resolve_floor_randomizer,
    resolve_table_randomizer,
)


@register_env("PickAnything-v1", max_episode_steps=50)
class PickAnythingEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda
    goal_thresh = 0.025

    def __init__(
        self,
        *args,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.02,
        num_envs: int = 1,
        reconfiguration_freq=None,
        object_sources: Optional[Sequence[Union[ObjectSource, str]]] = None,
        object_randomizer: Optional[Randomizer] = None,
        table_randomizer: Optional[Union[Randomizer, str, Sequence[str]]] = None,
        floor_randomizer: Optional[Union[Randomizer, str]] = None,
        clutter: Optional[Union[int, ClutterRandomizer]] = None,
        lighting_randomizer: Optional[Randomizer] = None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # the object source pool (reused by the clutter randomizer below).
        sources = list(object_sources) if object_sources else [
            "cube",
            "ycb",
            "interndata",
        ]
        # build randomizers before super().__init__ (they only hold config; they
        # touch the env later via their hooks).
        if object_randomizer is not None:
            self.object_randomizer = object_randomizer
        else:
            # interndata is a gated HF dataset --- it needs
            # `huggingface-cli login` + license acceptance and downloads meshes on
            # first use; pass object_sources=["cube","ycb"] for a no-download default.
            self.object_randomizer = CompositeObjectRandomizer(
                sources, goal_thresh=self.goal_thresh
            )
        # table: "wood" (fixed PickCube wood) / "texture" (random InternDataAssets
        # table textures + randomized friction) / "procedural" (PBR color) / a
        # Randomizer instance / a list of aliases to mix per reconfigure. Default
        # ["wood","texture"] mixes wood and textured tables per reconfigure.
        self.table_randomizer = resolve_table_randomizer(
            table_randomizer if table_randomizer is not None else ["wood", "texture"],
            robot_init_qpos_noise=robot_init_qpos_noise,
        )
        # floor: "texture" (default, InternDataAssets floor_textures +
        # background_textures) / "grid" (checkered, no download) / a Randomizer.
        self.floor_randomizer = resolve_floor_randomizer(
            floor_randomizer if floor_randomizer is not None else "texture"
        )
        # clutter: int N (N distractors per env, reusing the object source pool) /
        # a ClutterRandomizer / None (0 = no distractors, the default). Orthogonal
        # to the other axes; distractors are physical but not in the state obs.
        if clutter is None or (isinstance(clutter, int) and clutter <= 0):
            self.clutter_randomizer = None
        elif isinstance(clutter, int):
            self.clutter_randomizer = ClutterRandomizer(
                num_clutter=clutter, sources=sources
            )
        elif isinstance(clutter, ClutterRandomizer):
            self.clutter_randomizer = clutter
        else:
            raise TypeError(
                f"clutter must be an int, ClutterRandomizer, or None; got {clutter!r}"
            )
        self.lighting_randomizer = lighting_randomizer or HDRILightingRandomizer()
        if reconfiguration_freq is None:
            # single env: reconfigure (and thus re-randomize geometry) every
            # episode. many envs: opt-in via reconfiguration_freq>=1.
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Sensors
    # ------------------------------------------------------------------ #
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.7, 0.6], target=[0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    # ------------------------------------------------------------------ #
    # Scene / lighting / episode lifecycle (delegated to randomizers)
    # ------------------------------------------------------------------ #
    def _load_scene(self, options: dict):
        self.table_randomizer.on_reconfigure(self, options)
        self.floor_randomizer.on_reconfigure(self, options)
        self.object_randomizer.on_reconfigure(self, options)
        if self.clutter_randomizer is not None:
            self.clutter_randomizer.on_reconfigure(self, options)

    def _load_lighting(self, options: dict):
        self.lighting_randomizer.on_reconfigure(self, options)

    def _after_reconfigure(self, options: dict):
        # post-build measurement (e.g. object resting heights from collision
        # meshes) before the first episode initializes.
        self.table_randomizer.on_after_reconfigure(self, options)
        self.floor_randomizer.on_after_reconfigure(self, options)
        self.object_randomizer.on_after_reconfigure(self, options)
        if self.clutter_randomizer is not None:
            self.clutter_randomizer.on_after_reconfigure(self, options)
        self.lighting_randomizer.on_after_reconfigure(self, options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_randomizer.on_initialize_episode(self, env_idx, options)
        self.floor_randomizer.on_initialize_episode(self, env_idx, options)
        self.object_randomizer.on_initialize_episode(self, env_idx, options)
        # clutter placed after the target so it can avoid the target's pose
        if self.clutter_randomizer is not None:
            self.clutter_randomizer.on_initialize_episode(self, env_idx, options)
        self.lighting_randomizer.on_initialize_episode(self, env_idx, options)

        if self.robot_uids == "panda":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            b = len(env_idx)
            qpos = (
                self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            qpos[:, -2:] = 0.04
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    # ------------------------------------------------------------------ #
    # Task logic (same as PickCube / PickSingleYCB)
    # ------------------------------------------------------------------ #
    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.obj.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.obj)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.obj.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.obj.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
