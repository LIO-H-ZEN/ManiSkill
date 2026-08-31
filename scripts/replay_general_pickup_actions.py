#!/usr/bin/env python3
"""Replay one recorded General Pickup action sequence in ManiSkill.

The output uses a simulator-neutral telemetry schema shared with the RoboDojo
direct-action replay script.  State sample 0 is captured immediately after
reset; state sample i + 1 is captured after source action i has run for one
20 Hz control period.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.agents.robots.piper.piper_wristcam import PiperWristCam
from mani_skill.utils.visualization.misc import images_to_video, put_text_on_image
from scripts.general_pickup_direct_replay_common import (
    CONTROL_FREQUENCY_HZ,
    PHYSICS_FREQUENCY_HZ,
    SCHEMA_VERSION,
    gripper_action_to_joint7,
    load_source_episode,
    resolve_replay_action_count,
)

VIDEO_CAMERA_NAMES = ("base_camera", "wrist_camera", "side_camera")
VIDEO_CAMERA_LABELS = {
    "base_camera": "ManiSkill | Base camera",
    "wrist_camera": "ManiSkill | Wrist camera",
    "side_camera": "ManiSkill | Side camera",
}


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _single(value: Any, *, name: str) -> np.ndarray:
    result = _numpy(value)
    if result.ndim >= 1 and result.shape[0] == 1:
        result = result[0]
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def build_robot_self_contact_pairs(env: Any) -> list[tuple[str, Any, Any]]:
    links = list(env.unwrapped.agent.robot.links)
    names = [link.name for link in links]
    if len(names) != len(set(names)):
        raise ValueError(f"robot link names must be unique, got {names}")
    return [
        (f"{names[left]}::{names[right]}", links[left], links[right])
        for left in range(len(links))
        for right in range(left + 1, len(links))
    ]


def sample_state(
    env: Any,
    *,
    initial_target_z: float,
    robot_self_contact_pairs: list[tuple[str, Any, Any]] | None = None,
) -> dict[str, Any]:
    base = env.unwrapped
    qpos = _single(base.agent.robot.get_qpos(), name="qpos").astype(np.float32)
    tcp_pose = _single(base.agent.tcp.pose.raw_pose, name="tcp pose").astype(np.float32)
    left_finger_pose = _single(
        base.agent.finger1_link.pose.raw_pose, name="left finger pose"
    ).astype(np.float32)
    right_finger_pose = _single(
        base.agent.finger2_link.pose.raw_pose, name="right finger pose"
    ).astype(np.float32)
    target_pose = _single(base.obj.pose.raw_pose, name="target pose").astype(np.float32)
    left_force = _single(
        base.scene.get_pairwise_contact_forces(base.agent.finger1_link, base.obj),
        name="left contact force",
    )
    right_force = _single(
        base.scene.get_pairwise_contact_forces(base.agent.finger2_link, base.obj),
        name="right contact force",
    )
    evaluation = base.evaluate()
    state = {
        "qpos": qpos,
        "tcp_pose": tcp_pose,
        "left_finger_pose": left_finger_pose,
        "right_finger_pose": right_finger_pose,
        "target_pose": target_pose,
        "target_lift_height_m": np.float32(target_pose[2] - initial_target_z),
        "gripper_opening_m": np.float32(abs(float(qpos[6])) + abs(float(qpos[7]))),
        "left_contact_force_n": np.float32(np.linalg.norm(left_force)),
        "right_contact_force_n": np.float32(np.linalg.norm(right_force)),
        "is_grasped": bool(_single(evaluation["is_grasped"], name="is_grasped")),
        "legacy_success_3step": bool(
            _single(evaluation["legacy_success_3step"], name="legacy_success_3step")
        ),
        "robust_success_10step": bool(
            _single(evaluation["robust_success_10step"], name="robust_success_10step")
        ),
    }
    if robot_self_contact_pairs is not None:
        state["robot_self_contact_force_n"] = np.asarray(
            [
                np.linalg.norm(
                    _single(
                        base.scene.get_pairwise_contact_forces(left, right),
                        name=f"robot self contact {name}",
                    )
                )
                for name, left, right in robot_self_contact_pairs
            ],
            dtype=np.float32,
        )
    return state


def capture_frame(obs: dict[str, Any], *, prompt: str, step: int) -> np.ndarray:
    panels = []
    for camera_name in VIDEO_CAMERA_NAMES:
        try:
            frame = _single(
                obs["sensor_data"][camera_name]["rgb"],
                name=f"{camera_name} RGB",
            )
        except KeyError as error:
            raise RuntimeError(
                f"{camera_name} RGB is required for triptych replay video"
            ) from error
        if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
            raise ValueError(
                f"unexpected {camera_name} frame: {frame.dtype} {frame.shape}"
            )
        panels.append(
            put_text_on_image(
                frame,
                [VIDEO_CAMERA_LABELS[camera_name], f"step={step}", prompt],
            )
        )
    return np.concatenate(panels, axis=1)


def write_npz_atomic(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trajectory", type=Path, required=True)
    parser.add_argument("--contract-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", default="pair-0328-roll-03")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-backend", default="sapien_cuda")
    parser.add_argument(
        "--sim-backend", default="physx_cpu", choices=("physx_cpu", "physx_cuda")
    )
    parser.add_argument("--max-actions", type=int)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--disable-self-collisions", action="store_true")
    parser.add_argument("--record-robot-self-contacts", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = load_source_episode(args.source_trajectory, candidate_id=args.candidate_id)
    action_count = resolve_replay_action_count(
        total_actions=source["actions"].shape[0], max_actions=args.max_actions
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    PiperWristCam.disable_self_collisions = args.disable_self_collisions
    env = gym.make(
        "RoboDojoGeneralPickupPiper-v1",
        layout_id=source["layout_id"],
        contract_root=args.contract_root,
        asset_root=args.asset_root,
        obs_mode="state" if args.no_video else "rgb",
        render_mode=None if args.no_video else "rgb_array",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        num_envs=1,
    )
    states: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    robot_self_contact_pairs: list[tuple[str, Any, Any]] | None = None
    try:
        obs, _ = env.reset(seed=args.seed, options={"reconfigure": False})
        if args.record_robot_self_contacts:
            robot_self_contact_pairs = build_robot_self_contact_pairs(env)
        initial_target_z = float(
            _single(env.unwrapped.obj.pose.p, name="target position")[2]
        )
        states.append(
            sample_state(
                env,
                initial_target_z=initial_target_z,
                robot_self_contact_pairs=robot_self_contact_pairs,
            )
        )
        if not args.no_video:
            frames.append(capture_frame(obs, prompt=source["prompt"], step=0))
        for action_index, action in enumerate(source["actions"][:action_count]):
            obs, _, _, _, _ = env.step(action)
            states.append(
                sample_state(
                    env,
                    initial_target_z=initial_target_z,
                    robot_self_contact_pairs=robot_self_contact_pairs,
                )
            )
            if not args.no_video:
                frames.append(
                    capture_frame(obs, prompt=source["prompt"], step=action_index + 1)
                )
    finally:
        env.close()

    actions = source["actions"][:action_count]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "simulator": "ManiSkill",
        "layout_id": source["layout_id"],
        "candidate_id": args.candidate_id,
        "prompt": source["prompt"],
        "source_trajectory": str(args.source_trajectory),
        "source_trajectory_sha256": source["sha256"],
        "control_frequency_hz": CONTROL_FREQUENCY_HZ,
        "physics_frequency_hz": PHYSICS_FREQUENCY_HZ,
        "physics_steps_per_action": PHYSICS_FREQUENCY_HZ // CONTROL_FREQUENCY_HZ,
        "arm_command_semantics": "absolute_joint_position_radians",
        "source_gripper_semantics": "normalized_minus1_closed_plus1_open",
        "physical_gripper_semantics": "joint7_meters_0_closed_0.035_open",
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "steps": action_count,
        "source_steps": int(source["actions"].shape[0]),
        "video_enabled": not args.no_video,
        "video_camera_names": list(VIDEO_CAMERA_NAMES) if not args.no_video else [],
        "disable_self_collisions": args.disable_self_collisions,
        "robot_self_contacts_recorded": args.record_robot_self_contacts,
        "final_robust_success_10step": states[-1]["robust_success_10step"],
        "max_target_lift_height_m": max(
            float(state["target_lift_height_m"]) for state in states
        ),
    }
    arrays = {
        "state_time_s": np.arange(len(states), dtype=np.float64) / CONTROL_FREQUENCY_HZ,
        "action_time_s": np.arange(len(actions), dtype=np.float64)
        / CONTROL_FREQUENCY_HZ,
        "source_action": actions,
        "source_actual_qpos_pre_action": source["source_qpos"][:action_count],
        "source_expert_stage": source["stages"][:action_count],
        "commanded_arm_qpos": actions[:, :6],
        "commanded_gripper_normalized": actions[:, 6],
        "commanded_gripper_joint7_m": np.asarray(
            [gripper_action_to_joint7(value) for value in actions[:, 6]],
            dtype=np.float32,
        ),
        "qpos": np.stack([state["qpos"] for state in states]),
        "tcp_pose_wxyz": np.stack([state["tcp_pose"] for state in states]),
        "left_finger_pose_wxyz": np.stack(
            [state["left_finger_pose"] for state in states]
        ),
        "right_finger_pose_wxyz": np.stack(
            [state["right_finger_pose"] for state in states]
        ),
        "target_pose_wxyz": np.stack([state["target_pose"] for state in states]),
        "target_lift_height_m": np.asarray(
            [state["target_lift_height_m"] for state in states], dtype=np.float32
        ),
        "gripper_opening_m": np.asarray(
            [state["gripper_opening_m"] for state in states], dtype=np.float32
        ),
        "left_contact_force_n": np.asarray(
            [state["left_contact_force_n"] for state in states], dtype=np.float32
        ),
        "right_contact_force_n": np.asarray(
            [state["right_contact_force_n"] for state in states], dtype=np.float32
        ),
        "is_grasped": np.asarray(
            [state["is_grasped"] for state in states], dtype=np.bool_
        ),
        "legacy_success_3step": np.asarray(
            [state["legacy_success_3step"] for state in states], dtype=np.bool_
        ),
        "robust_success_10step": np.asarray(
            [state["robust_success_10step"] for state in states], dtype=np.bool_
        ),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if robot_self_contact_pairs is not None:
        arrays["robot_self_contact_pair_names"] = np.asarray(
            [name for name, _, _ in robot_self_contact_pairs]
        )
        arrays["robot_self_contact_force_n"] = np.stack(
            [state["robot_self_contact_force_n"] for state in states]
        )
    telemetry_path = args.output_dir / "maniskill_replay_telemetry.npz"
    write_npz_atomic(telemetry_path, arrays)
    if not args.no_video:
        images_to_video(
            frames,
            str(args.output_dir),
            f"maniskill_layout{source['layout_id']}_triptych",
            fps=CONTROL_FREQUENCY_HZ,
            verbose=False,
        )
    manifest_path = args.output_dir / "maniskill_replay_manifest.json"
    manifest_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {**metadata, "telemetry": str(telemetry_path), "video_frames": len(frames)},
            indent=2,
        )
    )
    return (
        0
        if action_count < source["actions"].shape[0]
        else (0 if metadata["final_robust_success_10step"] else 2)
    )


if __name__ == "__main__":
    raise SystemExit(main())
