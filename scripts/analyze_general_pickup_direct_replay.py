#!/usr/bin/env python3
"""Compare synchronized ManiSkill and RoboDojo direct-action replay traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "general_pickup_direct_action_replay_comparison_v2"
MANISKILL_PIPER_TCP_OFFSET_M = 0.1358


def quaternion_angle_rad(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.ndim != 2 or lhs.shape[1] != 4:
        raise ValueError(
            f"quaternion arrays must have matching shape (T, 4), got {lhs.shape} and {rhs.shape}"
        )
    lhs = lhs / np.linalg.norm(lhs, axis=1, keepdims=True)
    rhs = rhs / np.linalg.norm(rhs, axis=1, keepdims=True)
    dots = np.clip(np.abs(np.sum(lhs * rhs, axis=1)), 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.ndim != 2 or lhs.shape[1] != 4:
        raise ValueError(
            f"quaternion arrays must have matching shape (T, 4), got {lhs.shape} and {rhs.shape}"
        )
    lw, lx, ly, lz = lhs.T
    rw, rx, ry, rz = rhs.T
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=1,
    )


def quaternion_local_z_axis(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if (
        quaternion.ndim != 2
        or quaternion.shape[1] != 4
        or not np.isfinite(quaternion).all()
    ):
        raise ValueError(f"quaternions must be finite (T, 4), got {quaternion.shape}")
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("quaternions must be non-zero")
    w, x, y, z = (quaternion / norm).T
    return np.stack(
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=1,
    )


def translate_pose_along_local_z(
    pose_wxyz: np.ndarray, distance_m: float
) -> np.ndarray:
    pose = np.asarray(pose_wxyz, dtype=np.float64)
    distance = float(distance_m)
    if pose.ndim != 2 or pose.shape[1] != 7 or not np.isfinite(pose).all():
        raise ValueError(f"poses must be finite (T, 7), got {pose.shape}")
    if not np.isfinite(distance):
        raise ValueError(f"local-z translation must be finite, got {distance_m!r}")
    translated = pose.copy()
    translated[:, :3] += quaternion_local_z_axis(pose[:, 3:7]) * distance
    return translated


def build_tcp_frame_contract(static_contract: dict[str, Any]) -> dict[str, Any]:
    configs = static_contract.get("configs")
    if not isinstance(configs, dict):
        raise ValueError("layout contract is missing static_contract.configs")
    piper_asset = configs.get("piper_asset")
    if not isinstance(piper_asset, dict):
        raise ValueError(
            "layout contract is missing static_contract.configs.piper_asset"
        )
    if piper_asset.get("ee_link") != "link6":
        raise ValueError("RoboDojo Piper asset must use ee_link='link6'")
    robodojo_offset = piper_asset.get("gripper_bias")
    if (
        isinstance(robodojo_offset, bool)
        or not isinstance(robodojo_offset, (int, float))
        or not np.isfinite(robodojo_offset)
        or robodojo_offset <= 0.0
    ):
        raise ValueError("RoboDojo Piper gripper_bias must be a positive finite number")
    return {
        "maniskill_ee_frame": "piper_tcp",
        "maniskill_link6_to_tcp_m": MANISKILL_PIPER_TCP_OFFSET_M,
        "robodojo_ee_frame": "link6",
        "robodojo_policy_tcp_offset_m": float(robodojo_offset),
    }


def transform_maniskill_pose_to_robodojo(
    pose_wxyz: np.ndarray, coordinate_contract: dict[str, Any]
) -> np.ndarray:
    pose = np.asarray(pose_wxyz, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7 or not np.isfinite(pose).all():
        raise ValueError(f"poses must be finite (T, 7), got {pose.shape}")
    if coordinate_contract.get("quaternion_order") != "wxyz":
        raise ValueError("coordinate contract must use wxyz quaternions")
    if coordinate_contract.get("robodojo_R_maniskill") != "Rz(+90deg)":
        raise ValueError(
            "coordinate contract must use RoboDojo Rz(+90deg) ManiSkill transform"
        )
    mani_base = np.asarray(
        coordinate_contract.get("maniskill_piper_base_position"), dtype=np.float64
    )
    robo_base = np.asarray(
        coordinate_contract.get("robodojo_piper_base_position"), dtype=np.float64
    )
    if (
        mani_base.shape != (3,)
        or robo_base.shape != (3,)
        or not np.isfinite([mani_base, robo_base]).all()
    ):
        raise ValueError("coordinate contract base positions must be finite 3-vectors")

    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = pose.copy()
    transformed[:, :3] = (pose[:, :3] - mani_base) @ rotation.T + robo_base
    rz_quaternion = np.broadcast_to(
        np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]), pose[:, 3:7].shape
    )
    transformed[:, 3:7] = quaternion_multiply(rz_quaternion, pose[:, 3:7])
    return transformed


def first_exceedance(values: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(np.asarray(values) > threshold)
    return int(indices[0]) if indices.size else None


def summarize_tracking(
    values: np.ndarray, threshold: float, *, state_index_offset: int
) -> dict[str, Any]:
    error = np.asarray(values, dtype=np.float64)
    if error.ndim != 1 or error.size == 0 or not np.isfinite(error).all():
        raise ValueError("tracking error must be a non-empty finite vector")
    first = first_exceedance(error, threshold)
    return {
        "first_exceedance_state_index": None
        if first is None
        else first + state_index_offset,
        "max": float(np.max(error)),
    }


def load_trace(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as source:
        result = {key: np.asarray(source[key]) for key in source.files}
    metadata = json.loads(str(result.pop("metadata_json")))
    if metadata.get("schema_version") != "general_pickup_direct_action_replay_v1":
        raise ValueError(
            f"unsupported telemetry schema in {path}: {metadata.get('schema_version')!r}"
        )
    result["metadata"] = metadata
    return result


def compare_traces(
    mani: dict[str, Any],
    robo: dict[str, Any],
    *,
    coordinate_contract: dict[str, Any],
    tcp_frame_contract: dict[str, Any],
    arm_threshold_rad: float,
    gripper_threshold_m: float,
    tcp_position_threshold_m: float,
    tcp_orientation_threshold_deg: float,
    target_height_threshold_m: float,
) -> dict[str, Any]:
    if mani["source_action"].shape != robo["source_action"].shape:
        raise ValueError("source action shapes differ")
    if not np.array_equal(mani["source_action"], robo["source_action"]):
        raise ValueError("the two replays did not use bit-identical source actions")
    state_count = mani["qpos"].shape[0]
    if robo["qpos"].shape[0] != state_count:
        raise ValueError("state sample counts differ")
    action_count = mani["source_action"].shape[0]
    if state_count != action_count + 1:
        raise ValueError("telemetry must contain exactly one more state than action")
    for name, shape in {
        "source_actual_qpos_pre_action": (action_count, 8),
        "commanded_arm_qpos": (action_count, 6),
        "commanded_gripper_joint7_m": (action_count,),
    }.items():
        for simulator, trace in (("ManiSkill", mani), ("RoboDojo", robo)):
            values = np.asarray(trace[name])
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(
                    f"{simulator} {name} must be finite {shape}, got {values.shape}"
                )
        if not np.array_equal(mani[name], robo[name]):
            raise ValueError(f"the two replays did not use bit-identical {name}")
    if not np.array_equal(mani["commanded_arm_qpos"], mani["source_action"][:, :6]):
        raise ValueError("commanded arm qpos does not match the source action")

    expected_tcp_frame_contract = {
        "maniskill_ee_frame": "piper_tcp",
        "maniskill_link6_to_tcp_m": MANISKILL_PIPER_TCP_OFFSET_M,
        "robodojo_ee_frame": "link6",
    }
    for name, expected in expected_tcp_frame_contract.items():
        if tcp_frame_contract.get(name) != expected:
            raise ValueError(
                f"unexpected TCP frame contract {name}: {tcp_frame_contract.get(name)!r}"
            )
    robodojo_tcp_offset = tcp_frame_contract.get("robodojo_policy_tcp_offset_m")
    if (
        isinstance(robodojo_tcp_offset, bool)
        or not isinstance(robodojo_tcp_offset, (int, float))
        or not np.isfinite(robodojo_tcp_offset)
        or robodojo_tcp_offset <= 0.0
    ):
        raise ValueError("RoboDojo policy TCP offset must be a positive finite number")

    arm_error = np.max(np.abs(mani["qpos"][:, :6] - robo["qpos"][:, :6]), axis=1)
    gripper_error = np.abs(mani["gripper_opening_m"] - robo["gripper_opening_m"])
    mani_recorded_tcp_pose = transform_maniskill_pose_to_robodojo(
        mani["tcp_pose_wxyz"], coordinate_contract
    )
    robo_recorded_tcp_pose = np.asarray(robo["tcp_pose_wxyz"], dtype=np.float64)
    mani_link6_pose = translate_pose_along_local_z(
        mani_recorded_tcp_pose, -MANISKILL_PIPER_TCP_OFFSET_M
    )
    robo_link6_pose = translate_pose_along_local_z(
        robo_recorded_tcp_pose, -robodojo_tcp_offset
    )
    mani_structural_tcp_pose = translate_pose_along_local_z(
        mani_link6_pose, MANISKILL_PIPER_TCP_OFFSET_M
    )
    robo_structural_tcp_pose = translate_pose_along_local_z(
        robo_link6_pose, MANISKILL_PIPER_TCP_OFFSET_M
    )
    mani_target_pose = transform_maniskill_pose_to_robodojo(
        mani["target_pose_wxyz"], coordinate_contract
    )
    recorded_mixed_tcp_position_error = np.linalg.norm(
        mani_recorded_tcp_pose[:, :3] - robo_recorded_tcp_pose[:, :3], axis=1
    )
    link6_position_error = np.linalg.norm(
        mani_link6_pose[:, :3] - robo_link6_pose[:, :3], axis=1
    )
    structural_tcp_position_error = np.linalg.norm(
        mani_structural_tcp_pose[:, :3] - robo_structural_tcp_pose[:, :3], axis=1
    )
    tcp_orientation_error_deg = np.rad2deg(
        quaternion_angle_rad(mani_link6_pose[:, 3:7], robo_link6_pose[:, 3:7])
    )
    target_position_error = np.linalg.norm(
        mani_target_pose[:, :3] - robo["target_pose_wxyz"][:, :3], axis=1
    )
    target_height_error = np.abs(
        mani["target_lift_height_m"] - robo["target_lift_height_m"]
    )
    thresholds = {
        "arm_qpos_max_abs_rad": arm_threshold_rad,
        "gripper_opening_m": gripper_threshold_m,
        "link6_position_m": tcp_position_threshold_m,
        "structural_tcp_position_m": tcp_position_threshold_m,
        "tcp_orientation_deg": tcp_orientation_threshold_deg,
        "target_lift_height_m": target_height_threshold_m,
    }
    series = {
        "arm_qpos_max_abs_rad": arm_error,
        "gripper_opening_m": gripper_error,
        "link6_position_m": link6_position_error,
        "structural_tcp_position_m": structural_tcp_position_error,
        "tcp_orientation_deg": tcp_orientation_error_deg,
        "target_lift_height_m": target_height_error,
    }
    first = {
        name: first_exceedance(values, thresholds[name])
        for name, values in series.items()
    }
    finite_first = [value for value in first.values() if value is not None]
    first_any = min(finite_first) if finite_first else None
    mani_contact = (mani["left_contact_force_n"] > 0.5) & (
        mani["right_contact_force_n"] > 0.5
    )
    target_motion_threshold = 0.002
    mani_motion = np.abs(mani["target_lift_height_m"]) > target_motion_threshold
    robo_motion = np.abs(robo["target_lift_height_m"]) > target_motion_threshold
    motion_disagreement = np.flatnonzero(mani_motion != robo_motion)
    source_actual_tracking = {}
    command_arm_tracking = {}
    command_gripper_tracking = {}
    for simulator, trace in (("maniskill", mani), ("robodojo", robo)):
        source_arm_error = np.max(
            np.abs(
                trace["qpos"][:-1, :6] - trace["source_actual_qpos_pre_action"][:, :6]
            ),
            axis=1,
        )
        command_arm_error = np.max(
            np.abs(trace["qpos"][1:, :6] - trace["commanded_arm_qpos"]), axis=1
        )
        expected_opening = 2.0 * trace["commanded_gripper_joint7_m"]
        command_gripper_error = np.abs(
            trace["gripper_opening_m"][1:] - expected_opening
        )
        source_actual_tracking[simulator] = summarize_tracking(
            source_arm_error, arm_threshold_rad, state_index_offset=0
        )
        command_arm_tracking[simulator] = summarize_tracking(
            command_arm_error, arm_threshold_rad, state_index_offset=1
        )
        command_gripper_tracking[simulator] = summarize_tracking(
            command_gripper_error, gripper_threshold_m, state_index_offset=1
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": mani["metadata"]["candidate_id"],
        "layout_id": mani["metadata"]["layout_id"],
        "state_samples": state_count,
        "action_samples": action_count,
        "thresholds": thresholds,
        "first_exceedance_state_index": first,
        "first_any_divergence_state_index": first_any,
        "first_target_motion_disagreement_state_index": (
            int(motion_disagreement[0]) if motion_disagreement.size else None
        ),
        "maniskill_first_two_pad_contact_state_index": (
            int(np.flatnonzero(mani_contact)[0]) if np.any(mani_contact) else None
        ),
        "coordinate_contract": coordinate_contract,
        "tcp_frame_contract": tcp_frame_contract,
        "initial_alignment_errors": {
            "arm_qpos_max_abs_rad": float(arm_error[0]),
            "link6_position_m": float(link6_position_error[0]),
            "structural_tcp_position_m": float(structural_tcp_position_error[0]),
            "recorded_mixed_tcp_position_m": float(
                recorded_mixed_tcp_position_error[0]
            ),
            "tcp_orientation_deg": float(tcp_orientation_error_deg[0]),
            "target_position_m": float(target_position_error[0]),
        },
        "tcp_frame_diagnostics": {
            "recorded_frames_are_directly_comparable": False,
            "max_recorded_mixed_tcp_position_error_m": float(
                np.max(recorded_mixed_tcp_position_error)
            ),
        },
        "tracking_errors": {
            "source_actual_arm_qpos_max_abs_rad": source_actual_tracking,
            "post_action_command_arm_qpos_max_abs_rad": command_arm_tracking,
            "post_action_command_gripper_opening_m": command_gripper_tracking,
        },
        "max_errors": {name: float(np.max(values)) for name, values in series.items()},
        "final": {
            "maniskill_target_lift_height_m": float(mani["target_lift_height_m"][-1]),
            "robodojo_target_lift_height_m": float(robo["target_lift_height_m"][-1]),
            "maniskill_max_target_lift_height_m": float(
                np.max(mani["target_lift_height_m"])
            ),
            "robodojo_max_target_lift_height_m": float(
                np.max(robo["target_lift_height_m"])
            ),
            "maniskill_robust_success_10step": bool(mani["robust_success_10step"][-1]),
            "robodojo_robust_success_10step": bool(robo["robust_success_10step"][-1]),
        },
    }
    if first_any is None and summary["final"]["robodojo_robust_success_10step"]:
        conclusion = "same-actions-agree-and-succeed-visual-domain-gap-likely"
    elif source_actual_tracking["robodojo"]["first_exceedance_state_index"] is not None:
        conclusion = "control-or-physics-contract-divergence"
    elif (
        first["arm_qpos_max_abs_rad"] is not None
        or first["link6_position_m"] is not None
    ):
        conclusion = "executed-robot-trajectory-divergence"
    elif (
        summary["final"]["maniskill_robust_success_10step"]
        and not summary["final"]["robodojo_robust_success_10step"]
    ):
        conclusion = "contact-or-object-physics-divergence"
    else:
        conclusion = "inconclusive"
    summary["conclusion"] = conclusion
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maniskill", type=Path, required=True)
    parser.add_argument("--robodojo", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm-threshold-rad", type=float, default=0.05)
    parser.add_argument("--gripper-threshold-m", type=float, default=0.005)
    parser.add_argument("--tcp-position-threshold-m", type=float, default=0.02)
    parser.add_argument("--tcp-orientation-threshold-deg", type=float, default=10.0)
    parser.add_argument("--target-height-threshold-m", type=float, default=0.005)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    mani = load_trace(args.maniskill)
    robo = load_trace(args.robodojo)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    expected_contract_sha = robo["metadata"].get("layout_contract_sha256")
    if contract.get("contract_sha256") != expected_contract_sha:
        raise ValueError(
            f"RoboDojo telemetry contract hash mismatch: expected {expected_contract_sha!r}, "
            f"got {contract.get('contract_sha256')!r}"
        )
    static_contract = contract.get("static_contract", {})
    coordinate_contract = static_contract.get("coordinate_contract")
    if not isinstance(coordinate_contract, dict):
        raise ValueError(
            "layout contract is missing static_contract.coordinate_contract"
        )
    tcp_frame_contract = build_tcp_frame_contract(static_contract)
    summary = compare_traces(
        mani,
        robo,
        coordinate_contract=coordinate_contract,
        tcp_frame_contract=tcp_frame_contract,
        arm_threshold_rad=args.arm_threshold_rad,
        gripper_threshold_m=args.gripper_threshold_m,
        tcp_position_threshold_m=args.tcp_position_threshold_m,
        tcp_orientation_threshold_deg=args.tcp_orientation_threshold_deg,
        target_height_threshold_m=args.target_height_threshold_m,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
