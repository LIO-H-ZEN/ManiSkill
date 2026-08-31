#!/usr/bin/env python3

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.examples.motionplanning.piper.grasping.contracts import (
    GraspProviderName,
)
from mani_skill.examples.motionplanning.piper.solutions.lift_anything import (
    DEFAULT_GRASP_PROVIDER_CASCADE,
    ExpertResult,
    solve,
    solve_provider_cascade,
)

CAMERA_UIDS = {
    "front": "base_camera",
    "wrist": "wrist_camera",
    "side": "side_camera",
}
GRIPPER_JOINT_LIMIT = 0.035
CASCADE_PROVIDER_REQUEST = "cascade"
PROVIDER_REQUEST_CHOICES = (
    CASCADE_PROVIDER_REQUEST,
    *(item.value for item in GraspProviderName),
)


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def parse_layout_ids(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items or any(item < 0 for item in items):
        raise ValueError("layout IDs must be a non-empty list of non-negative integers")
    if len(items) != len(set(items)):
        raise ValueError("layout IDs must be unique")
    return items


def gripper_action_to_joint7(action: float) -> np.float32:
    value = float(action)
    if not np.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(f"gripper action must be finite and in [-1, 1], got {action}")
    return np.float32((value + 1.0) * 0.5 * GRIPPER_JOINT_LIMIT)


class GeneralPickupRecorder(gym.Wrapper):
    """Record the observation and previous gripper command before each action."""

    def __init__(self, env):
        super().__init__(env)
        self.expert_stage = "reset"
        self._current_obs = None
        self._commanded_gripper_joint7 = np.float32(0.0)
        self.frames: list[dict[str, Any]] = []
        self.max_lift_height = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        qpos = _to_numpy(self.unwrapped.agent.robot.get_qpos())[0]
        if qpos.shape != (8,):
            raise ValueError(f"expected PiPER qpos shape (8,), got {qpos.shape}")
        self._commanded_gripper_joint7 = np.float32(qpos[6])
        self._current_obs = obs
        self.frames = []
        self.expert_stage = "reset"
        self.max_lift_height = 0.0
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError(f"invalid PiPER action: {action}")
        if not self.action_space.contains(action):
            raise ValueError(f"PiPER expert action would be clipped: {action}")
        if self._current_obs is None:
            raise RuntimeError("recorder step called before reset")

        qpos = _to_numpy(self.unwrapped.agent.robot.get_qpos())[0]
        sensor_data = self._current_obs["sensor_data"]
        images = {}
        for output_name, camera_uid in CAMERA_UIDS.items():
            image = _to_numpy(sensor_data[camera_uid]["rgb"])[0]
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"{camera_uid} RGB must be uint8 (224, 224, 3), "
                    f"got {image.dtype} {image.shape}"
                )
            images[output_name] = image.copy()

        state = np.concatenate(
            [qpos[:6], np.asarray([self._commanded_gripper_joint7])]
        ).astype(np.float32)
        self.frames.append(
            {
                "images": images,
                "state": state,
                "actual_qpos": qpos.astype(np.float32, copy=True),
                "action": action.copy(),
                "expert_stage": self.expert_stage,
            }
        )
        transition = self.env.step(action)
        self._commanded_gripper_joint7 = gripper_action_to_joint7(action[6])
        if "lift_height" in transition[4]:
            self.max_lift_height = max(
                self.max_lift_height,
                float(_to_numpy(transition[4]["lift_height"]).reshape(-1)[0]),
            )
        self._current_obs = transition[0]
        return transition


@dataclasses.dataclass(frozen=True)
class AttemptResult:
    layout_id: int
    episode_id: str
    target_asset_key: str
    prompt: str
    accepted: bool
    reason: str
    candidate_id: str | None
    attempted_candidates: int
    steps: int
    max_lift_height: float
    robust_success_10step: bool
    shard_path: str | None
    shard_sha256: str | None
    provider_request: str
    selected_provider: str | None
    candidate_evaluations: tuple[dict[str, Any], ...] = ()


def solve_provider_request(
    env,
    *,
    provider_request: str,
    seed: int,
    grasp_cache_dir: Path | None,
    candidate_pool_limit: int,
) -> ExpertResult:
    if provider_request == CASCADE_PROVIDER_REQUEST:
        return solve_provider_cascade(
            env,
            seed=seed,
            grasp_cache_dir=grasp_cache_dir,
            candidate_pool_limit=candidate_pool_limit,
        )
    provider = GraspProviderName(provider_request)
    return solve(
        env,
        seed=seed,
        provider=provider,
        pipeline="common",
        grasp_cache_dir=grasp_cache_dir,
        candidate_pool_limit=candidate_pool_limit,
    )


def selected_provider(result: ExpertResult) -> str | None:
    if not result.success:
        return None
    matches = {
        evaluation.provider.value
        for evaluation in result.evaluations
        if evaluation.candidate_id == result.candidate_id
        and evaluation.robust_success_10step
    }
    if len(matches) != 1:
        raise RuntimeError(
            "accepted expert result must identify one unique successful provider"
        )
    return matches.pop()


def maximum_attempt_lift_height(
    result: ExpertResult, recorder_max_lift_height: float
) -> float:
    heights = [float(recorder_max_lift_height)]
    heights.extend(float(item.max_lift_height) for item in result.evaluations)
    if any(not np.isfinite(value) or value < 0.0 for value in heights):
        raise ValueError("attempt lift heights must be finite and non-negative")
    return max(heights)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(_canonical_json_bytes(payload))
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_episode(
    *,
    output_dir: Path,
    episode_id: str,
    frames: list[dict[str, Any]],
    prompt: str,
    layout_spec: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[Path, str]:
    if not frames:
        raise ValueError("cannot write an empty expert episode")
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    final_path = episode_dir / f"{episode_id}.npz"
    if final_path.exists():
        raise FileExistsError(final_path)
    arrays = {
        **{
            f"observation.images.{name}": np.stack(
                [frame["images"][name] for frame in frames]
            )
            for name in CAMERA_UIDS
        },
        "observation.state": np.stack([frame["state"] for frame in frames]),
        "observation.actual_qpos": np.stack([frame["actual_qpos"] for frame in frames]),
        "action_command_raw": np.stack([frame["action"] for frame in frames]),
        "action_command_applied": np.stack([frame["action"] for frame in frames]),
        "expert_stage": np.asarray(
            [frame["expert_stage"] for frame in frames], dtype="U16"
        ),
        "prompt": np.asarray(prompt),
        "layout_spec_json": np.asarray(json.dumps(layout_spec, sort_keys=True)),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    with tempfile.NamedTemporaryFile(
        dir=episode_dir, prefix=f".{episode_id}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, final_path)
    finally:
        temporary.unlink(missing_ok=True)
    return final_path, _sha256_file(final_path)


def collect_layout(
    *,
    layout_id: int,
    contract_root: Path,
    asset_root: Path,
    grasp_cache_dir: Path,
    output_dir: Path,
    render_backend: str,
    candidate_pool_limit: int,
    seed: int,
    provider: str = GraspProviderName.ANTIPODAL.value,
) -> AttemptResult:
    if provider not in PROVIDER_REQUEST_CHOICES:
        raise ValueError(f"unsupported provider request: {provider}")
    episode_id = f"general-pickup-layout-{layout_id:03d}-seed-{seed:06d}"
    contract_path = contract_root / f"general_pickup_filtered_{layout_id:03d}.json"
    layout_spec = json.loads(contract_path.read_text(encoding="utf-8"))
    env = gym.make(
        "RoboDojoGeneralPickupPiper-v1",
        layout_id=layout_id,
        contract_root=contract_root,
        asset_root=asset_root,
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        render_backend=render_backend,
        num_envs=1,
    )
    recorder = GeneralPickupRecorder(env)
    try:
        result = solve_provider_request(
            recorder,
            provider_request=provider,
            seed=seed,
            grasp_cache_dir=grasp_cache_dir,
            candidate_pool_limit=candidate_pool_limit,
        )
        selected = selected_provider(result)
        evaluations = tuple(item.to_dict() for item in result.evaluations)
        info = {} if result.transition is None else result.transition[4]
        robust_success = bool(
            _to_numpy(info.get("robust_success_10step", False)).reshape(-1)[0]
        )
        max_lift_height = maximum_attempt_lift_height(
            result, recorder.max_lift_height
        )
        target_asset_key = env.unwrapped.layout_spec.target.asset_key
        prompt = env.unwrapped.instruction
        if not result.success or not robust_success:
            return AttemptResult(
                layout_id,
                episode_id,
                target_asset_key,
                prompt,
                False,
                result.reason,
                result.candidate_id,
                result.attempted_candidates,
                len(recorder.frames),
                max_lift_height,
                robust_success,
                None,
                None,
                provider,
                selected,
                evaluations,
            )
        metadata = {
            "layout_id": layout_id,
            "episode_id": episode_id,
            "target_asset_key": target_asset_key,
            "prompt": prompt,
            "candidate_id": result.candidate_id,
            "attempted_candidates": result.attempted_candidates,
            "steps": len(recorder.frames),
            "max_lift_height": max_lift_height,
            "robust_success_10step": robust_success,
            "camera_contract": "piper_sim_camera_v1",
            "control_mode": "robodojo_pd_joint_pos",
            "state_gripper_semantics": "previous_commanded_joint7_meters",
            "provider": selected,
            "provider_request": provider,
            "selected_provider": selected,
            "provider_order": (
                [item.value for item in DEFAULT_GRASP_PROVIDER_CASCADE]
                if provider == CASCADE_PROVIDER_REQUEST
                else [provider]
            ),
            "pipeline": "common",
            "candidate_evaluations": evaluations,
        }
        shard_path, digest = _write_episode(
            output_dir=output_dir,
            episode_id=episode_id,
            frames=recorder.frames,
            prompt=prompt,
            layout_spec=layout_spec,
            metadata=metadata,
        )
        return AttemptResult(
            layout_id,
            episode_id,
            target_asset_key,
            prompt,
            True,
            "accepted",
            result.candidate_id,
            result.attempted_candidates,
            len(recorder.frames),
            recorder.max_lift_height,
            robust_success,
            str(shard_path),
            digest,
            provider,
            selected,
            evaluations,
        )
    finally:
        recorder.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-ids", default="9")
    parser.add_argument("--contract-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--grasp-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-backend", default="cuda:0")
    parser.add_argument("--candidate-pool-limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--provider",
        choices=PROVIDER_REQUEST_CHOICES,
        default=GraspProviderName.ANTIPODAL.value,
    )
    args = parser.parse_args()
    if args.candidate_pool_limit <= 0:
        raise ValueError("candidate-pool-limit must be positive")
    layout_ids = parse_layout_ids(args.layout_ids)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    results = []
    for layout_id in layout_ids:
        result = collect_layout(
            layout_id=layout_id,
            contract_root=args.contract_root.resolve(),
            asset_root=args.asset_root.resolve(),
            grasp_cache_dir=args.grasp_cache_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            render_backend=args.render_backend,
            candidate_pool_limit=args.candidate_pool_limit,
            seed=args.seed,
            provider=args.provider,
        )
        results.append(result)
        print(json.dumps(dataclasses.asdict(result), sort_keys=True), flush=True)

    manifest = {
        "contract_manifest_sha256": _sha256_file(args.contract_root / "manifest.json"),
        "conversion_manifest_sha256": _sha256_file(args.asset_root / "manifest.json"),
        "attempted": len(results),
        "accepted": sum(result.accepted for result in results),
        "provider": args.provider,
        "pipeline": "common",
        "results": [dataclasses.asdict(result) for result in results],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    _write_json_atomic(args.output_dir / "manifest.json", manifest)
    if not all(result.accepted for result in results):
        raise RuntimeError("one or more General Pickup expert episodes failed")


if __name__ == "__main__":
    main()
