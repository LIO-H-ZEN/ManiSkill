"""Object randomizers for PickAnything.

v1 ships a single procedural randomizer: a box with random color and random
(graspable) size, built per parallel env. v2/v3 will add YCB and InternDataAssets
mesh variants behind the same ``on_reconfigure`` / ``on_initialize_episode``
interface.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import sapien
import torch

from mani_skill.utils.building import actors
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

import mani_skill.envs.utils.randomization as randomization

from .base import Randomizer


class ProceduralObjectRandomizer(Randomizer):
    """A pickable box with randomized color and size.

    Geometry/color are sampled and built at reconfigure time (they are baked
    into the scene graph); pose and goal are randomized per episode.
    """

    def __init__(
        self,
        goal_thresh: float = 0.025,
        half_size_range=(0.015, 0.03),
        spawn_half_size: float = 0.1,
        spawn_center=(0.0, 0.0),
        max_goal_height: float = 0.3,
    ):
        self.goal_thresh = goal_thresh
        self.half_size_range = half_size_range
        self.spawn_half_size = spawn_half_size
        self.spawn_center = spawn_center
        self.max_goal_height = max_goal_height

    def on_reconfigure(self, env, options: dict) -> None:
        b = env.num_envs
        rng = env._batched_episode_rng

        # one scalar per env -> shape (num_envs,)
        half_sizes = rng.uniform(*self.half_size_range)
        # one rgb triple per env -> shape (num_envs, 3)
        colors = rng.uniform(0.1, 1.0, size=(3,))
        colors = np.concatenate([colors, np.ones((b, 1))], axis=1)  # add alpha

        objs: list[Actor] = []
        for i in range(b):
            hs = float(half_sizes[i])
            builder = env.scene.create_actor_builder()
            builder.add_box_collision(half_size=[hs] * 3)
            builder.add_box_visual(
                half_size=[hs] * 3,
                material=sapien.render.RenderMaterial(base_color=colors[i].tolist()),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, hs])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"object-{i}")
            env.remove_from_state_dict_registry(obj)
            objs.append(obj)

        env._objs = objs
        env.obj = Actor.merge(objs, name="object") if b > 1 else objs[0]
        env.add_to_state_dict_registry(env.obj)
        # box bottom sits on the table (z=0) when placed at z=half_size, so the
        # per-env spawn height is just the per-env half size.
        env.object_half_sizes = torch.tensor(half_sizes, device=env.device, dtype=torch.float32)

        env.goal_site = actors.build_sphere(
            env.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        env._hidden_objects.append(env.goal_site)

    def on_initialize_episode(self, env, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(env.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.spawn_half_size * 2 - self.spawn_half_size
            )
            xyz[:, 0] += self.spawn_center[0]
            xyz[:, 1] += self.spawn_center[1]
            xyz[:, 2] = env.object_half_sizes[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            env.obj.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.spawn_half_size * 2 - self.spawn_half_size
            )
            goal_xyz[:, 0] += self.spawn_center[0]
            goal_xyz[:, 1] += self.spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b,)) * self.max_goal_height + xyz[:, 2]
            env.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
