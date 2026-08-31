"""Pure helpers for General Pickup direct-action replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "general_pickup_direct_action_replay_v1"
CONTROL_FREQUENCY_HZ = 20
PHYSICS_FREQUENCY_HZ = 200
GRIPPER_JOINT_LIMIT_M = 0.035


def resolve_replay_action_count(*, total_actions: int, max_actions: int | None) -> int:
    total = int(total_actions)
    if total <= 0:
        raise ValueError(f"total_actions must be positive, got {total_actions!r}")
    if max_actions is None:
        return total
    if (
        isinstance(max_actions, bool)
        or not isinstance(max_actions, int)
        or max_actions <= 0
    ):
        raise ValueError(f"max_actions must be a positive integer, got {max_actions!r}")
    if max_actions > total:
        raise ValueError(
            f"max_actions {max_actions} exceeds source action count {total}"
        )
    return max_actions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def gripper_action_to_joint7(action: float) -> float:
    value = float(action)
    if not np.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(f"gripper action must be finite and in [-1, 1], got {action}")
    return (value + 1.0) * 0.5 * GRIPPER_JOINT_LIMIT_M


def load_source_episode(path: Path, *, candidate_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as source:
        required = {
            "action_command_applied",
            "observation.actual_qpos",
            "expert_stage",
            "metadata_json",
            "prompt",
        }
        missing = sorted(required.difference(source.files))
        if missing:
            raise ValueError(f"source trajectory is missing arrays: {missing}")
        actions = np.asarray(source["action_command_applied"], dtype=np.float32)
        source_qpos = np.asarray(source["observation.actual_qpos"], dtype=np.float32)
        stages = np.asarray(source["expert_stage"])
        metadata = json.loads(str(source["metadata_json"]))
        prompt = str(source["prompt"])
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f"source actions must be finite (T, 7), got {actions.shape}")
    if source_qpos.shape != (actions.shape[0], 8) or not np.isfinite(source_qpos).all():
        raise ValueError(
            f"source actual qpos must be finite {(actions.shape[0], 8)}, got {source_qpos.shape}"
        )
    if stages.shape != (actions.shape[0],):
        raise ValueError(
            f"source expert stages must have shape {(actions.shape[0],)}, got {stages.shape}"
        )
    if np.any(actions[:, 6] < -1.0) or np.any(actions[:, 6] > 1.0):
        raise ValueError("source gripper commands leave [-1, 1]")
    if metadata.get("candidate_id") != candidate_id:
        raise ValueError(
            f"expected candidate {candidate_id!r}, got {metadata.get('candidate_id')!r}"
        )
    layout_id = metadata.get("layout_id")
    if isinstance(layout_id, bool) or not isinstance(layout_id, int) or layout_id < 0:
        raise ValueError(f"layout_id must be a non-negative integer, got {layout_id!r}")
    if metadata.get("control_mode") != "robodojo_pd_joint_pos":
        raise ValueError(
            f"unexpected source control mode: {metadata.get('control_mode')!r}"
        )
    if not metadata.get("robust_success_10step"):
        raise ValueError("source trajectory is not robust_success_10step")
    return {
        "actions": actions,
        "source_qpos": source_qpos,
        "stages": stages,
        "metadata": metadata,
        "layout_id": layout_id,
        "prompt": prompt,
        "sha256": sha256_file(path),
    }
