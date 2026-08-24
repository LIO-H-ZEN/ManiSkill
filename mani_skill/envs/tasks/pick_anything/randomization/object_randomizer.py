"""Object randomizers for PickAnything.

Two randomizers ship here:

* :class:`ProceduralObjectRandomizer` --- a single box with random color/size
  per env (the original v1 randomizer; no assets, handy for smoke tests).

* :class:`CompositeObjectRandomizer` --- the main one. It holds a list of
  :class:`~mani_skill.envs.tasks.pick_anything.randomization.object_sources.ObjectSource`
  (cube / YCB / InternDataAssets / ...) and, each reconfiguration, samples one
  source per parallel env. Object resting heights are measured generically from
  each actor's collision mesh in ``on_after_reconfigure`` (the PickSingleYCB
  trick), so any source --- box, YCB mesh, downloaded mesh --- places itself
  flat on the table without knowing its own geometry.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import sapien
import torch

from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

import mani_skill.envs.utils.randomization as randomization

from .base import Randomizer
from .object_sources import ObjectSource, resolve_object_source


# ---------------------------------------------------------------------------- #
# Composite: mix multiple object sources
# ---------------------------------------------------------------------------- #
class CompositeObjectRandomizer(Randomizer):
    """Sample one object source per env per reconfiguration and build it.

    Args:
        sources: sequence of :class:`ObjectSource` instances or string aliases
            (``"cube"``, ``"ycb"``, ``"interndata"``). One source is drawn per
            parallel env at each reconfigure, so a run can mix sources freely.
        goal_thresh: radius of the (hidden) green goal sphere.
        spawn_half_size: object/goal xy are sampled in
            ``[center - s, center + s]``.
        spawn_center: xy center of the spawn region.
        max_goal_height: goal z is sampled in ``[obj_z, obj_z + h]``.
    """

    def __init__(
        self,
        sources: Sequence[Union[ObjectSource, str]],
        goal_thresh: float = 0.025,
        spawn_half_size: float = 0.1,
        spawn_center=(0.0, 0.0),
        max_goal_height: float = 0.3,
        create_goal: bool = True,
    ):
        if len(sources) == 0:
            raise ValueError("CompositeObjectRandomizer needs at least one source")
        self.sources: list[ObjectSource] = [resolve_object_source(s) for s in sources]
        self.goal_thresh = goal_thresh
        self.spawn_half_size = spawn_half_size
        self.spawn_center = spawn_center
        self.max_goal_height = max_goal_height
        self.create_goal = create_goal

    def on_reconfigure(self, env, options: dict) -> None:
        b = env.num_envs
        rng = env._batched_episode_rng

        # one source index per env -> shape (num_envs,)
        source_idx = rng.randint(0, len(self.sources))
        env.object_sources = [self.sources[int(i)].name for i in source_idx]

        objs: list[Actor] = []
        for i in range(b):
            src = self.sources[int(source_idx[i])]
            # hand the source this env's own sub-RNG so draws are deterministic
            # per env per seed and don't bleed across envs.
            obj = src.build_actor(env, i, rng[i])
            env.remove_from_state_dict_registry(obj)
            objs.append(obj)

        env._objs = objs
        env.obj = Actor.merge(objs, name="object") if b > 1 else objs[0]
        env.add_to_state_dict_registry(env.obj)

        if self.create_goal:
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

    def on_after_reconfigure(self, env, options: dict) -> None:
        # Resting height = distance from actor origin to the bottom of its
        # collision mesh. Works for boxes, YCB, and arbitrary downloaded meshes.
        zs = []
        for obj in env._objs:
            mesh = obj.get_first_collision_mesh()
            zs.append(-mesh.bounding_box.bounds[0, 2])
        env.object_zs = common.to_tensor(zs, device=env.device)

    def on_initialize_episode(self, env, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(env.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.spawn_half_size * 2 - self.spawn_half_size
            )
            xyz[:, 0] += self.spawn_center[0]
            xyz[:, 1] += self.spawn_center[1]
            xyz[:, 2] = env.object_zs[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            env.obj.set_pose(Pose.create_from_pq(xyz, qs))

            if self.create_goal:
                goal_xyz = torch.zeros((b, 3))
                goal_xyz[:, :2] = (
                    torch.rand((b, 2)) * self.spawn_half_size * 2
                    - self.spawn_half_size
                )
                goal_xyz[:, 0] += self.spawn_center[0]
                goal_xyz[:, 1] += self.spawn_center[1]
                goal_xyz[:, 2] = (
                    torch.rand((b,)) * self.max_goal_height + xyz[:, 2]
                )
                env.goal_site.set_pose(Pose.create_from_pq(goal_xyz))


# ---------------------------------------------------------------------------- #
# Procedural box (no assets; kept for quick smoke tests / backward compat)
# ---------------------------------------------------------------------------- #
class ProceduralObjectRandomizer(Randomizer):
    """A pickable box with randomized color and size.

    Geometry/color are sampled and built at reconfigure time (baked into the
    scene graph); pose and goal are randomized per episode.
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

        half_sizes = rng.uniform(*self.half_size_range)
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
            builder.initial_pose = sapien.Pose()
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"object-{i}")
            env.remove_from_state_dict_registry(obj)
            objs.append(obj)

        env._objs = objs
        env.obj = Actor.merge(objs, name="object") if b > 1 else objs[0]
        env.add_to_state_dict_registry(env.obj)
        env.object_half_sizes = torch.tensor(
            half_sizes, device=env.device, dtype=torch.float32
        )

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
