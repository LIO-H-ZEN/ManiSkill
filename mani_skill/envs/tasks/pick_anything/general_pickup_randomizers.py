"""Exact-layout randomizers for RoboDojo General Pickup replay."""

from __future__ import annotations

from pathlib import Path

import sapien
import torch

from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor

from .general_pickup_specs import RoboDojoLayoutSpec, RoboDojoSceneObjectSpec
from .randomization.base import Randomizer
from .randomization.object_sources import RoboDojoConvertedObjectSource


def _source(
    spec: RoboDojoSceneObjectSpec, asset_root: str | Path
) -> RoboDojoConvertedObjectSource:
    return RoboDojoConvertedObjectSource(
        asset_type=spec.asset_type,
        category=spec.category,
        object_id=spec.object_id,
        asset_root=asset_root,
        scale=spec.scale,
        mass=spec.mass,
        friction=spec.friction,
    )


class RoboDojoExactTargetRandomizer(Randomizer):
    """Build and place the one target encoded by a layout contract."""

    def __init__(self, layout: RoboDojoLayoutSpec, asset_root: str | Path):
        self.layout = layout
        self.source = _source(layout.target, asset_root)

    def on_reconfigure(self, env, options: dict) -> None:
        if env.num_envs != 1:
            raise ValueError("RoboDojo exact layout replay requires num_envs=1")
        actor = self.source.build_actor(
            env,
            0,
            env._batched_episode_rng[0],
            name=f"target-{self.layout.target.category}-{self.layout.target.object_id}",
        )
        env.remove_from_state_dict_registry(actor)
        env._objs = [actor]
        env.obj = actor
        env.object_sources = ["robodojo"]
        env.object_names = [actor.name]
        env.add_to_state_dict_registry(actor)

    def on_after_reconfigure(self, env, options: dict) -> None:
        mesh = env.obj.get_first_collision_mesh(to_world_frame=False)
        if mesh is None:
            raise RuntimeError("RoboDojo target has no collision mesh")
        env.object_zs = common.to_tensor(
            [-mesh.bounding_box.bounds[0, 2]], device=env.device
        )

    def on_initialize_episode(self, env, env_idx: torch.Tensor, options: dict) -> None:
        if env_idx.tolist() != [0]:
            raise ValueError("RoboDojo exact target reset requires env_idx=[0]")
        env.obj.set_pose(
            sapien.Pose(
                p=self.layout.target.position,
                q=self.layout.target.quaternion,
            )
        )
        env.obj.set_linear_velocity([0.0, 0.0, 0.0])
        env.obj.set_angular_velocity([0.0, 0.0, 0.0])


class RoboDojoExactClutterRandomizer(Randomizer):
    """Build and place every distractor from the same RoboDojo layout."""

    def __init__(self, layout: RoboDojoLayoutSpec, asset_root: str | Path):
        self.layout = layout
        self.sources = tuple(_source(spec, asset_root) for spec in layout.clutter)

    def on_reconfigure(self, env, options: dict) -> None:
        if env.num_envs != 1:
            raise ValueError("RoboDojo exact layout replay requires num_envs=1")
        actors = []
        for index, (spec, source) in enumerate(zip(self.layout.clutter, self.sources)):
            actor = source.build_actor(
                env,
                0,
                env._batched_episode_rng[0],
                name=f"clutter-{index}-{spec.category}-{spec.object_id}",
            )
            env.remove_from_state_dict_registry(actor)
            actors.append(actor)
        env._clutter_objs = actors
        env.clutter_objs = (
            Actor.merge(actors, name="clutter") if len(actors) > 1 else actors[0]
        )
        env.clutter_count = (len(actors), len(actors))

    def on_after_reconfigure(self, env, options: dict) -> None:
        heights = []
        for actor in env._clutter_objs:
            mesh = actor.get_first_collision_mesh(to_world_frame=False)
            if mesh is None:
                raise RuntimeError(
                    f"RoboDojo clutter has no collision mesh: {actor.name}"
                )
            heights.append(-mesh.bounding_box.bounds[0, 2])
        env.clutter_zs = common.to_tensor(heights, device=env.device)

    def on_initialize_episode(self, env, env_idx: torch.Tensor, options: dict) -> None:
        if env_idx.tolist() != [0]:
            raise ValueError("RoboDojo exact clutter reset requires env_idx=[0]")
        for actor, spec in zip(env._clutter_objs, self.layout.clutter):
            actor.set_pose(sapien.Pose(p=spec.position, q=spec.quaternion))
            actor.set_linear_velocity([0.0, 0.0, 0.0])
            actor.set_angular_velocity([0.0, 0.0, 0.0])

    def on_step(self, env, env_idx: torch.Tensor, options: dict) -> None:
        raise RuntimeError(
            "RoboDojo exact layout clutter cannot be randomized in-episode"
        )
