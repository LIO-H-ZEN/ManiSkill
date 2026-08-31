import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
REPLAY_SCRIPT = ROOT / "scripts/replay_general_pickup_actions.py"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


replay = _load(
    "general_pickup_direct_replay_common",
    "scripts/general_pickup_direct_replay_common.py",
)
analysis = _load(
    "general_pickup_direct_replay_analysis",
    "scripts/analyze_general_pickup_direct_replay.py",
)

COORDINATE_CONTRACT = {
    "maniskill_piper_base_position": [-0.35, 0.0, 0.0],
    "quaternion_order": "wxyz",
    "robodojo_R_maniskill": "Rz(+90deg)",
    "robodojo_piper_base_position": [0.0, -0.45, 0.765],
}
TCP_FRAME_CONTRACT = {
    "maniskill_ee_frame": "piper_tcp",
    "maniskill_link6_to_tcp_m": 0.1358,
    "robodojo_ee_frame": "link6",
    "robodojo_policy_tcp_offset_m": 0.18,
}


def test_replay_video_uses_locked_three_camera_order():
    tree = ast.parse(REPLAY_SCRIPT.read_text(encoding="utf-8"))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "VIDEO_CAMERA_NAMES"
    }
    assert assignments == {
        "VIDEO_CAMERA_NAMES": ("base_camera", "wrist_camera", "side_camera")
    }
    capture = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "capture_frame"
    )
    calls = [
        ast.unparse(node) for node in ast.walk(capture) if isinstance(node, ast.Call)
    ]
    assert "np.concatenate(panels, axis=1)" in calls


def test_gripper_action_mapping():
    assert replay.gripper_action_to_joint7(-1.0) == pytest.approx(0.0)
    assert replay.gripper_action_to_joint7(0.0) == pytest.approx(0.0175)
    assert replay.gripper_action_to_joint7(1.0) == pytest.approx(0.035)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        replay.gripper_action_to_joint7(1.01)


def test_resolve_replay_action_count_supports_strict_prefixes():
    assert (
        replay.resolve_replay_action_count(total_actions=134, max_actions=None) == 134
    )
    assert replay.resolve_replay_action_count(total_actions=134, max_actions=10) == 10
    with pytest.raises(ValueError, match="positive"):
        replay.resolve_replay_action_count(total_actions=134, max_actions=0)
    with pytest.raises(ValueError, match="exceeds"):
        replay.resolve_replay_action_count(total_actions=134, max_actions=135)


@pytest.mark.parametrize("layout_id", [0, 2007])
def test_load_source_episode_accepts_and_returns_nonnegative_layout_id(
    tmp_path, layout_id
):
    path = tmp_path / "episode.npz"
    np.savez_compressed(
        path,
        action_command_applied=np.zeros((2, 7), dtype=np.float32),
        **{
            "observation.actual_qpos": np.zeros((2, 8), dtype=np.float32),
            "expert_stage": np.asarray(["approach", "lift"]),
            "metadata_json": np.asarray(
                json.dumps(
                    {
                        "candidate_id": "pair-1704-roll-04",
                        "control_mode": "robodojo_pd_joint_pos",
                        "layout_id": layout_id,
                        "robust_success_10step": True,
                    }
                )
            ),
            "prompt": np.asarray("Pick up the wooden owl figurine by 10 cm."),
        },
    )

    source = replay.load_source_episode(path, candidate_id="pair-1704-roll-04")

    assert source["layout_id"] == layout_id


@pytest.mark.parametrize("layout_id", [True, -1, "2007"])
def test_load_source_episode_rejects_invalid_layout_id(tmp_path, layout_id):
    path = tmp_path / "episode.npz"
    np.savez_compressed(
        path,
        action_command_applied=np.zeros((1, 7), dtype=np.float32),
        **{
            "observation.actual_qpos": np.zeros((1, 8), dtype=np.float32),
            "expert_stage": np.asarray(["lift"]),
            "metadata_json": np.asarray(
                json.dumps(
                    {
                        "candidate_id": "candidate",
                        "control_mode": "robodojo_pd_joint_pos",
                        "layout_id": layout_id,
                        "robust_success_10step": True,
                    }
                )
            ),
            "prompt": np.asarray("Pick up the target by 10 cm."),
        },
    )

    with pytest.raises(ValueError, match="layout_id must be a non-negative integer"):
        replay.load_source_episode(path, candidate_id="candidate")


def test_quaternion_angle_is_sign_invariant():
    lhs = np.asarray([[1.0, 0.0, 0.0, 0.0]])
    rhs = -lhs
    np.testing.assert_allclose(analysis.quaternion_angle_rad(lhs, rhs), 0.0)


def test_first_exceedance_is_strict():
    values = np.asarray([0.0, 0.1, 0.10001])
    assert analysis.first_exceedance(values, 0.1) == 2
    assert analysis.first_exceedance(values, 1.0) is None


def test_compare_transforms_maniskill_world_pose_to_robodojo_world():
    mani = _trace()
    robo = _trace()
    # Both traces describe the same link6 pose, but the recorded TCP frames
    # differ: ManiSkill uses link6 + 0.1358 m and RoboDojo uses link6 + 0.18 m.
    mani["tcp_pose_wxyz"][:, :3] = [0.15, 0.1, 0.3358]
    mani["target_pose_wxyz"][:, :3] = [-0.1, -0.2, 0.03]
    robo["tcp_pose_wxyz"][:, :3] = [-0.1, 0.05, 1.145]
    robo["tcp_pose_wxyz"][:, 3:7] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    robo["target_pose_wxyz"][:, :3] = [0.2, -0.2, 0.795]
    robo["target_pose_wxyz"][:, 3:7] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]

    summary = _compare(mani, robo)

    assert summary["first_exceedance_state_index"]["link6_position_m"] is None
    assert summary["first_exceedance_state_index"]["structural_tcp_position_m"] is None
    assert summary["first_exceedance_state_index"]["tcp_orientation_deg"] is None
    assert summary["tcp_frame_diagnostics"][
        "max_recorded_mixed_tcp_position_error_m"
    ] == pytest.approx(0.0442)


def test_build_tcp_frame_contract_is_strict():
    static_contract = {
        "configs": {"piper_asset": {"ee_link": "link6", "gripper_bias": 0.18}}
    }
    assert analysis.build_tcp_frame_contract(static_contract) == TCP_FRAME_CONTRACT

    static_contract["configs"]["piper_asset"]["ee_link"] = "piper_tcp"
    with pytest.raises(ValueError, match="ee_link='link6'"):
        analysis.build_tcp_frame_contract(static_contract)


def test_compare_reports_source_and_command_tracking_separately():
    mani = _trace()
    robo = _trace()
    mani["commanded_arm_qpos"][0, 0] = 0.1
    robo["commanded_arm_qpos"][0, 0] = 0.1
    mani["source_action"][0, 0] = 0.1
    robo["source_action"][0, 0] = 0.1
    mani["qpos"][1, 0] = 0.2
    robo["qpos"][1, 0] = 0.1

    summary = _compare(mani, robo)

    command_tracking = summary["tracking_errors"][
        "post_action_command_arm_qpos_max_abs_rad"
    ]
    assert command_tracking["maniskill"]["first_exceedance_state_index"] == 1
    assert command_tracking["robodojo"]["first_exceedance_state_index"] is None
    source_tracking = summary["tracking_errors"]["source_actual_arm_qpos_max_abs_rad"]
    assert source_tracking["maniskill"]["max"] == 0.0
    assert source_tracking["robodojo"]["max"] == 0.0


def _trace():
    return {
        "source_action": np.zeros((1, 7), dtype=np.float32),
        "source_actual_qpos_pre_action": np.zeros((1, 8), dtype=np.float32),
        "commanded_arm_qpos": np.zeros((1, 6), dtype=np.float32),
        "commanded_gripper_joint7_m": np.zeros(1, dtype=np.float32),
        "qpos": np.zeros((2, 8), dtype=np.float32),
        "tcp_pose_wxyz": np.asarray([[0, 0, 0, 1, 0, 0, 0]] * 2, dtype=np.float32),
        "target_pose_wxyz": np.asarray([[0, 0, 0, 1, 0, 0, 0]] * 2, dtype=np.float32),
        "target_lift_height_m": np.zeros(2, dtype=np.float32),
        "gripper_opening_m": np.zeros(2, dtype=np.float32),
        "left_contact_force_n": np.zeros(2, dtype=np.float32),
        "right_contact_force_n": np.zeros(2, dtype=np.float32),
        "robust_success_10step": np.zeros(2, dtype=np.bool_),
        "metadata": {"candidate_id": "c", "layout_id": 9},
    }


def _compare(mani, robo):
    return analysis.compare_traces(
        mani,
        robo,
        coordinate_contract=COORDINATE_CONTRACT,
        tcp_frame_contract=TCP_FRAME_CONTRACT,
        arm_threshold_rad=0.05,
        gripper_threshold_m=0.005,
        tcp_position_threshold_m=0.02,
        tcp_orientation_threshold_deg=10.0,
        target_height_threshold_m=0.005,
    )


def test_compare_requires_identical_actions():
    base = _trace()
    other = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in base.items()
    }
    other["source_action"][0, 0] = 1.0
    with pytest.raises(ValueError, match="bit-identical"):
        _compare(base, other)
