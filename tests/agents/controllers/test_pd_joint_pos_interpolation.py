import torch

from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosController,
    PDJointPosControllerConfig,
)


def test_interpolation_steps_default_preserves_existing_behavior():
    config = PDJointPosControllerConfig(
        joint_names=["joint"],
        lower=None,
        upper=None,
        stiffness=100.0,
        damping=5.0,
        interpolate=True,
    )

    assert config.interpolation_steps is None


def test_piper_robodojo_controller_uses_eight_interpolation_steps():
    from mani_skill.agents.robots.piper.piper import Piper

    configs = Piper._controller_configs.fget(object.__new__(Piper))
    controller = configs["robodojo_pd_joint_pos"]

    assert controller["arm"].interpolate is True
    assert controller["arm"].interpolation_steps == 8
    assert controller["gripper"].interpolate is True
    assert controller["gripper"].interpolation_steps == 8
    assert controller["gripper"].stiffness == 500
    assert controller["gripper"].force_limit == 40


def test_piper_motion_planner_accepts_robodojo_position_control() -> None:
    from mani_skill.examples.motionplanning.piper.motionplanner import (
        PiperMotionPlanningSolver,
    )

    assert "robodojo_pd_joint_pos" in (
        PiperMotionPlanningSolver.SUPPORTED_CONTROL_MODES
    )


def test_interpolation_holds_final_target_after_configured_steps():
    controller = object.__new__(PDJointPosController)
    controller.config = type("Config", (), {"interpolate": True})()
    controller._step = 0
    controller._interpolation_steps = 8
    controller._start_qpos = torch.tensor([[0.0]])
    controller._step_size = torch.tensor([[0.125]])
    targets = []
    controller.set_drive_targets = lambda value: targets.append(float(value.item()))

    for _ in range(10):
        controller.before_simulation_step()

    assert targets == [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.0, 1.0]
