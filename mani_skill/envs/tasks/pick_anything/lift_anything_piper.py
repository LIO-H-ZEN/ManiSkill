"""Single-target arbitrary-object lift task for the PIPER embodiment."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.piper import Piper, PiperWristCam
from mani_skill.agents.robots.piper.piper_wristcam import PIPER_CAMERA_INTRINSIC
from mani_skill.envs.tasks.tabletop.lift_cube_piper import (
    LIFT_HEIGHT,
    PIPER_UIDS,
    SUCCESS_STREAK_STEPS,
    update_success_streak,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .episode_specs import EpisodeSpec, ObjectSpec
from .pick_anything_env import PickAnythingEnv
from .randomization import ClutterRandomizer, ObjectSource, Randomizer
from .randomization.object_randomizer import CompositeObjectRandomizer
from .randomization.object_sources import FixedObjectSource

SETTLING_PHYSICS_STEPS = 50
MAX_SETTLING_XY_DISPLACEMENT = 0.15
MAX_TABLE_PENETRATION = 0.003
MAX_PLANAR_REACH = 0.42


def validate_settled_spawn(
    *,
    initial_position: np.ndarray,
    settled_position: np.ndarray,
    settled_bottom_z: float,
    robot_base_position: np.ndarray,
) -> None:
    initial_position = np.asarray(initial_position, dtype=np.float64)
    settled_position = np.asarray(settled_position, dtype=np.float64)
    robot_base_position = np.asarray(robot_base_position, dtype=np.float64)
    if initial_position.shape != (3,) or settled_position.shape != (3,):
        raise ValueError("Object positions must have shape (3,)")
    if not np.all(np.isfinite(settled_position)):
        raise RuntimeError("spawn-invalid: object pose is non-finite after settling")
    if not np.isfinite(settled_bottom_z):
        raise RuntimeError("spawn-invalid: object bottom is non-finite after settling")
    if settled_bottom_z < -MAX_TABLE_PENETRATION:
        raise RuntimeError(
            f"spawn-invalid: object bottom penetrated table by {-settled_bottom_z:.6f} m"
        )
    xy_displacement = np.linalg.norm(settled_position[:2] - initial_position[:2])
    if xy_displacement > MAX_SETTLING_XY_DISPLACEMENT:
        raise RuntimeError(
            f"spawn-invalid: object moved {xy_displacement:.6f} m while settling"
        )
    planar_reach = np.linalg.norm(settled_position[:2] - robot_base_position[:2])
    if planar_reach > MAX_PLANAR_REACH:
        raise RuntimeError(
            f"spawn-invalid: planar reach {planar_reach:.6f} exceeds {MAX_PLANAR_REACH} m"
        )


@register_env("LiftAnythingPiper-v1", max_episode_steps=100)
class LiftAnythingPiperEnv(PickAnythingEnv):
    """Grasp one arbitrary object and lift it 10 cm for three control steps."""

    SUPPORTED_ROBOTS = ["piper", "piper_wristcam"]
    agent: Piper | PiperWristCam

    def __init__(
        self,
        *args,
        robot_uids: str = "piper_wristcam",
        robot_init_qpos_noise: float = 0.02,
        num_envs: int = 1,
        reconfiguration_freq=None,
        domain_rand_freq: int = 25,
        domain_rand_axes: Sequence[str] | None = None,
        episode_spec: EpisodeSpec | Mapping[str, Any] | None = None,
        object_spec: ObjectSpec | Mapping[str, Any] | None = None,
        object_sources: Sequence[Union[ObjectSource, str]] | None = None,
        object_randomizer: Randomizer | None = None,
        table_randomizer: Randomizer | str | Sequence[str] | None = None,
        floor_randomizer: Randomizer | str | None = None,
        clutter: int | str | ClutterRandomizer | None = None,
        clutter_sources: Sequence[Union[ObjectSource, str]] | None = None,
        lighting_randomizer: Randomizer | None = None,
        **kwargs,
    ):
        if robot_uids not in PIPER_UIDS:
            raise ValueError(
                f"LiftAnythingPiper-v1 requires one of {sorted(PIPER_UIDS)}, got {robot_uids!r}"
            )
        if episode_spec is not None and object_spec is not None:
            raise ValueError("episode_spec and object_spec are mutually exclusive")
        if (
            episode_spec is not None or object_spec is not None
        ) and object_randomizer is not None:
            raise ValueError(
                "fixed object contracts and object_randomizer are mutually exclusive"
            )
        if episode_spec is not None and num_envs != 1:
            raise ValueError("episode_spec requires num_envs=1")
        self.episode_spec = (
            EpisodeSpec.from_dict(episode_spec)
            if isinstance(episode_spec, Mapping)
            else episode_spec
        )
        self.fixed_object_spec = (
            ObjectSpec.from_dict(object_spec)
            if isinstance(object_spec, Mapping)
            else object_spec
        )
        if self.episode_spec is not None:
            self.fixed_object_spec = self.episode_spec.object_spec
        resolved_clutter_sources = clutter_sources
        if self.fixed_object_spec is not None:
            fixed_source = FixedObjectSource(self.fixed_object_spec)
            object_randomizer = CompositeObjectRandomizer(
                [fixed_source],
                spawn_half_size=0.06,
                spawn_center=(0.03, 0.0),
                create_goal=False,
            )
            if resolved_clutter_sources is None:
                resolved_clutter_sources = (fixed_source,)
        elif object_randomizer is None:
            object_randomizer = CompositeObjectRandomizer(
                object_sources or ("cube", "ycb", "interndata"),
                spawn_half_size=0.06,
                spawn_center=(0.03, 0.0),
                create_goal=False,
            )
        # BaseEnv performs one framework bootstrap reset before callers can
        # provide their episode seed. Validate only subsequent benchmark resets.
        self._in_constructor_reset = True
        try:
            super().__init__(
                *args,
                robot_uids=robot_uids,
                robot_init_qpos_noise=robot_init_qpos_noise,
                num_envs=num_envs,
                reconfiguration_freq=reconfiguration_freq,
                object_randomizer=object_randomizer,
                object_sources=object_sources,
                table_randomizer=table_randomizer,
                floor_randomizer=floor_randomizer,
                clutter=clutter,
                clutter_sources=resolved_clutter_sources,
                lighting_randomizer=lighting_randomizer,
                domain_rand_freq=domain_rand_freq,
                domain_rand_axes=domain_rand_axes,
                **kwargs,
            )
        finally:
            self._in_constructor_reset = False

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
                "base_camera",
                base_pose,
                224,
                224,
                None,
                0.01,
                100,
                intrinsic=PIPER_CAMERA_INTRINSIC.copy(),
            ),
            CameraConfig(
                "side_camera",
                side_pose,
                224,
                224,
                None,
                0.01,
                100,
                intrinsic=PIPER_CAMERA_INTRINSIC.copy(),
            ),
        ]

    def _load_agent(self, options: dict):
        super(PickAnythingEnv, self)._load_agent(
            options, sapien.Pose(p=[-0.35, 0.0, 0.0])
        )

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.object_rest_z = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.success_streak = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.streak_updated_at = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_randomizer.on_initialize_episode(self, env_idx, options)
        self.floor_randomizer.on_initialize_episode(self, env_idx, options)
        self.object_randomizer.on_initialize_episode(self, env_idx, options)
        if self.clutter_randomizer is not None:
            self.clutter_randomizer.on_initialize_episode(self, env_idx, options)
        self.lighting_randomizer.on_initialize_episode(self, env_idx, options)

        if self.episode_spec is None:
            qpos = np.repeat(
                self.agent.keyframes["home"].qpos[None, :], len(env_idx), axis=0
            )
            qpos[:, :6] += self._batched_episode_rng[env_idx].normal(
                0.0, self.robot_init_qpos_noise, size=(6,)
            )
        else:
            qpos = np.asarray([self.episode_spec.robot_init_qpos], dtype=np.float64)
            self.obj.set_pose(
                Pose.create_from_pq(
                    np.asarray([self.episode_spec.object_position]),
                    np.asarray([self.episode_spec.object_quaternion]),
                )
            )
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.35, 0.0, 0.0]))

        if self.num_envs == 1:
            initial_position = self.obj.pose.p[0].detach().cpu().numpy().copy()
            initial_quaternion = self.obj.pose.q[0].detach().cpu().numpy().copy()
            self._initial_robot_qpos = np.asarray(qpos[0], dtype=np.float64).copy()
            self._initial_object_position = initial_position
            self._initial_object_quaternion = initial_quaternion
            for _ in range(SETTLING_PHYSICS_STEPS):
                self.scene.step()
            settled_position = self.obj.pose.p[0].detach().cpu().numpy().copy()
            mesh = self.obj.get_first_collision_mesh(to_world_frame=True)
            if mesh is None:
                raise RuntimeError("spawn-invalid: object has no collision mesh")
            settled_bottom_z = float(mesh.bounding_box.bounds[0, 2])
            self._clear_sim_state()
            if not self._in_constructor_reset:
                validate_settled_spawn(
                    initial_position=initial_position,
                    settled_position=settled_position,
                    settled_bottom_z=settled_bottom_z,
                    robot_base_position=np.asarray(self.agent.robot.pose.p[0].cpu()),
                )
        self.object_rest_z[env_idx] = self.obj.pose.p[env_idx, 2]
        self.success_streak[env_idx] = 0
        self.streak_updated_at[env_idx] = 0
        self._elapsed_steps[env_idx] = 0
        if (
            self.episode_spec is not None
            and int(self._episode_seed[0]) == self.episode_spec.environment_seed
        ):
            self._assert_replay_metadata(self.episode_spec)

    def _realized_metadata(self) -> dict[str, Any]:
        table_kind = getattr(
            self,
            "table_randomizer_choice",
            type(self.table_randomizer).__name__,
        )
        hdri_files = getattr(self, "hdri_files", [None])
        return {
            "table_kind": table_kind,
            "table_texture": getattr(self, "table_texture", None),
            "table_friction": getattr(self, "table_friction", None),
            "floor_texture": getattr(self, "floor_texture", None),
            "hdri": hdri_files[0],
            "directional_light_direction": tuple(
                np.asarray(self.directional_light_direction).tolist()
            ),
            "directional_light_intensity": float(self.directional_light_intensity),
        }

    def _assert_replay_metadata(self, spec: EpisodeSpec) -> None:
        realized = self._realized_metadata()
        for key in ("table_kind", "table_texture", "floor_texture", "hdri"):
            if realized[key] != getattr(spec, key):
                raise RuntimeError(
                    f"EpisodeSpec replay mismatch for {key}: expected "
                    f"{getattr(spec, key)!r}, got {realized[key]!r}"
                )
        for key in (
            "table_friction",
            "directional_light_direction",
        ):
            expected = getattr(spec, key)
            actual = realized[key]
            if expected is None or actual is None:
                if expected != actual:
                    raise RuntimeError(
                        f"EpisodeSpec replay mismatch for {key}: expected {expected}, got {actual}"
                    )
            elif not np.allclose(actual, expected, atol=1e-6, rtol=0.0):
                raise RuntimeError(
                    f"EpisodeSpec replay mismatch for {key}: expected {expected}, got {actual}"
                )
        if not np.isclose(
            realized["directional_light_intensity"],
            spec.directional_light_intensity,
            atol=1e-6,
            rtol=0.0,
        ):
            raise RuntimeError("EpisodeSpec replay mismatch for light intensity")
        for key, actual, expected in (
            ("object_position", self._initial_object_position, spec.object_position),
            (
                "object_quaternion",
                self._initial_object_quaternion,
                spec.object_quaternion,
            ),
            ("robot_init_qpos", self._initial_robot_qpos, spec.robot_init_qpos),
        ):
            if not np.allclose(actual, expected, atol=1e-6, rtol=0.0):
                raise RuntimeError(
                    f"EpisodeSpec replay mismatch for {key}: expected {expected}, got {actual}"
                )

    def capture_episode_spec(self, stable_episode_id: str) -> EpisodeSpec:
        if self.num_envs != 1:
            raise RuntimeError("capture_episode_spec requires num_envs=1")
        if self.fixed_object_spec is None:
            raise RuntimeError(
                "capture_episode_spec requires a fixed ObjectSpec; random sources do not expose stable identity"
            )
        metadata = self._realized_metadata()
        return EpisodeSpec(
            stable_episode_id=stable_episode_id,
            environment_seed=int(self._episode_seed[0]),
            object_spec=self.fixed_object_spec,
            object_position=tuple(self._initial_object_position.tolist()),
            object_quaternion=tuple(self._initial_object_quaternion.tolist()),
            robot_init_qpos=tuple(self._initial_robot_qpos.tolist()),
            **metadata,
        )

    def _get_obs_extra(self, info: dict):
        return {}

    def evaluate(self):
        lift_height = self.obj.pose.p[:, 2] - self.object_rest_z
        is_lifted = lift_height >= LIFT_HEIGHT
        is_grasped = self.agent.is_grasping(self.obj)
        update_success_streak(
            self.success_streak,
            self.streak_updated_at,
            self._elapsed_steps,
            is_lifted & is_grasped,
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
            self.obj.pose.p - self.agent.tcp_pose.p, axis=1
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
        return self.compute_dense_reward(obs, action, info) / 5.0
