from types import SimpleNamespace

import numpy as np
import torch

from mani_skill.agents.robots.piper.piper_wristcam import (
    PIPER_CAMERA_INTRINSIC,
    build_piper_wrist_camera_config,
)
from mani_skill.envs.tasks.tabletop.lift_cube_piper import (
    LIFT_HEIGHT,
    SUCCESS_STREAK_STEPS,
    LiftCubePiperEnv,
    update_success_streak,
)


def test_camera_contract_is_exact() -> None:
    env = object.__new__(LiftCubePiperEnv)
    configs = {config.uid: config for config in env._default_sensor_configs}
    configs["wrist_camera"] = build_piper_wrist_camera_config(mount=object())

    assert set(configs) == {"base_camera", "side_camera", "wrist_camera"}
    for config in configs.values():
        assert config.width == 224
        assert config.height == 224
        assert config.fov is None
        np.testing.assert_allclose(config.intrinsic, PIPER_CAMERA_INTRINSIC)


def test_success_streak_updates_once_per_control_step() -> None:
    streak = torch.zeros(2, dtype=torch.int32)
    updated_at = torch.zeros(2, dtype=torch.int32)
    elapsed = torch.zeros(2, dtype=torch.int32)
    instantaneous = torch.tensor([True, False])

    update_success_streak(streak, updated_at, elapsed, instantaneous)
    torch.testing.assert_close(streak, torch.tensor([0, 0], dtype=torch.int32))

    for step in range(1, SUCCESS_STREAK_STEPS + 1):
        elapsed[0] = step
        update_success_streak(streak, updated_at, elapsed, instantaneous)
        update_success_streak(streak, updated_at, elapsed, instantaneous)
        assert streak[0].item() == step


def test_success_streak_clears_and_partial_reset_is_isolated() -> None:
    streak = torch.tensor([2, 2, 2], dtype=torch.int32)
    updated_at = torch.tensor([2, 2, 2], dtype=torch.int32)
    elapsed = torch.tensor([3, 2, 3], dtype=torch.int32)
    instantaneous = torch.tensor([True, False, False])

    update_success_streak(streak, updated_at, elapsed, instantaneous)

    torch.testing.assert_close(streak, torch.tensor([3, 2, 0], dtype=torch.int32))
    torch.testing.assert_close(updated_at, torch.tensor([3, 2, 3], dtype=torch.int32))


def test_evaluate_uses_relative_height_and_grasp() -> None:
    env = object.__new__(LiftCubePiperEnv)
    env.cube_rest_z = torch.tensor([0.02, 0.04])
    env.cube = SimpleNamespace(
        pose=SimpleNamespace(
            p=torch.tensor(
                [
                    [0.0, 0.0, 0.02 + LIFT_HEIGHT + 1e-4],
                    [0.0, 0.0, 0.20],
                ]
            )
        )
    )
    env.agent = SimpleNamespace(is_grasping=lambda cube: torch.tensor([True, False]))
    env.success_streak = torch.tensor([2, 2], dtype=torch.int32)
    env.streak_updated_at = torch.tensor([2, 2], dtype=torch.int32)
    env._elapsed_steps = torch.tensor([3, 3], dtype=torch.int32)

    info = env.evaluate()

    torch.testing.assert_close(info["is_lifted_10cm"], torch.tensor([True, True]))
    torch.testing.assert_close(info["success"], torch.tensor([True, False]))
    torch.testing.assert_close(
        info["success_streak"], torch.tensor([3, 0], dtype=torch.int32)
    )


def test_liftcube_has_no_goal_observation() -> None:
    env = object.__new__(LiftCubePiperEnv)
    assert env._get_obs_extra({"is_grasped": torch.tensor([True])}) == {}


def test_wristcam_uid_is_a_piper_table_robot() -> None:
    from mani_skill.utils.scene_builder.table.scene_builder import PIPER_TABLE_ROBOTS

    assert PIPER_TABLE_ROBOTS == {"piper", "piper_wristcam"}


def test_liftcube_horizon_is_gate_validated() -> None:
    from mani_skill.utils.registration import REGISTERED_ENVS

    assert REGISTERED_ENVS["LiftCubePiper-v1"].max_episode_steps == 100
