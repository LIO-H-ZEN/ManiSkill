"""RoboDojo General Pickup layouts replayed with the ManiSkill PiPER robot."""

from __future__ import annotations

from pathlib import Path

import torch

from mani_skill.envs.tasks.tabletop.lift_cube_piper import (
    LIFT_HEIGHT,
    SUCCESS_STREAK_STEPS,
    update_success_streak,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig

from .general_pickup_randomizers import (
    RoboDojoExactClutterRandomizer,
    RoboDojoExactTargetRandomizer,
)
from .general_pickup_specs import RoboDojoLayoutSpec, load_robodojo_layout_spec
from .lift_anything_piper import LiftAnythingPiperEnv
from .randomization import HDRILightingRandomizer
from .randomization.object_sources import resolve_robodojo_asset_root


@register_env("RoboDojoGeneralPickupPiper-v1", max_episode_steps=200)
class RoboDojoGeneralPickupPiperEnv(LiftAnythingPiperEnv):
    """Exact RoboDojo layout replay for transfer and expert smoke tests."""

    def __init__(
        self,
        *args,
        layout_id: int = 0,
        contract_root: str | Path | None = None,
        asset_root: str | Path | None = None,
        robot_uids: str = "piper_wristcam",
        control_mode: str = "robodojo_pd_joint_pos",
        num_envs: int = 1,
        **kwargs,
    ):
        if num_envs != 1:
            raise ValueError("RoboDojo exact layout replay requires num_envs=1")
        if control_mode != "robodojo_pd_joint_pos":
            raise ValueError(
                "RoboDojoGeneralPickupPiper-v1 requires "
                "control_mode='robodojo_pd_joint_pos'"
            )
        self.robodojo_asset_root = resolve_robodojo_asset_root(asset_root)
        self.layout_spec: RoboDojoLayoutSpec = load_robodojo_layout_spec(
            layout_id=layout_id,
            contract_root=contract_root,
            asset_root=self.robodojo_asset_root,
        )
        self.layout_id = self.layout_spec.filtered_layout_id
        self.instruction = self.layout_spec.prompt
        self.expert_reconfigure_each_candidate = False
        target_randomizer = RoboDojoExactTargetRandomizer(
            self.layout_spec, self.robodojo_asset_root
        )
        clutter_randomizer = RoboDojoExactClutterRandomizer(
            self.layout_spec, self.robodojo_asset_root
        )
        super().__init__(
            *args,
            robot_uids=robot_uids,
            control_mode=control_mode,
            num_envs=num_envs,
            robot_init_qpos_noise=0.0,
            reconfiguration_freq=0,
            domain_rand_freq=0,
            domain_rand_axes=(),
            object_randomizer=target_randomizer,
            clutter=clutter_randomizer,
            table_randomizer="wood",
            floor_randomizer="grid",
            lighting_randomizer=HDRILightingRandomizer(hdri_files=[]),
            maximum_planar_reach=None,
            success_streak_steps=10,
            **kwargs,
        )
        self.fixed_object_spec = self.layout_spec.target.object_spec

    @property
    def _default_sim_config(self) -> SimConfig:
        return SimConfig(sim_freq=200, control_freq=20)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.settled_clutter_states = tuple(
            actor.get_state()[0].detach().cpu().numpy().copy()
            for actor in self._clutter_objs
        )
        if not hasattr(self, "_cached_settled_target_state"):
            self._cached_settled_target_state = (
                self.obj.get_state()[0].detach().cpu().numpy().copy()
            )
            self._cached_settled_clutter_states = self.settled_clutter_states

    def _settle_scene(self) -> None:
        if not hasattr(self, "_cached_settled_target_state"):
            super()._settle_scene()
            return
        self.obj.set_state(self._cached_settled_target_state[None, :])
        for actor, state in zip(
            self._clutter_objs, self._cached_settled_clutter_states
        ):
            actor.set_state(state[None, :])

    def evaluate(self):
        lift_height = self.obj.pose.p[:, 2] - self.object_rest_z
        is_lifted = lift_height > LIFT_HEIGHT
        is_grasped = self.agent.is_grasping(self.obj)
        update_success_streak(
            self.success_streak,
            self.streak_updated_at,
            self._elapsed_steps,
            is_lifted & is_grasped,
        )
        return {
            "success": self.success_streak >= self.success_streak_steps,
            "robodojo_success": is_lifted,
            "legacy_success_3step": self.success_streak >= SUCCESS_STREAK_STEPS,
            "robust_success_10step": self.success_streak >= 10,
            "is_lifted_10cm": is_lifted,
            "is_grasped": is_grasped,
            "lift_height": lift_height,
            "success_streak": self.success_streak.clone(),
        }
