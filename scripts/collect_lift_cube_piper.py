#!/usr/bin/env python3

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import multiprocessing
import os
import pathlib
import tempfile
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.examples.motionplanning.piper.solutions.lift_cube import solve


CAMERA_UIDS = ("base_camera", "wrist_camera", "side_camera")
PROMPT = "pick up the red cube"


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class StrictEpisodeRecorder(gym.Wrapper):
    """Record policy frames before applying each unclipped controller action."""

    def __init__(self, env):
        super().__init__(env)
        self.expert_stage = "reset"
        self._current_obs = None
        self.frames: list[dict[str, Any]] = []
        self.max_lift_height = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        self.frames = []
        self.expert_stage = "reset"
        self.max_lift_height = 0.0
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError(
                f"Invalid PIPER action: shape={action.shape}, action={action}"
            )
        if not self.action_space.contains(action):
            raise ValueError(f"PIPER expert action would be clipped: {action}")
        if self._current_obs is None:
            raise RuntimeError("Recorder step called before reset")

        qpos = _to_numpy(self.unwrapped.agent.robot.get_qpos())[0]
        state = np.concatenate([qpos[:6], qpos[6:7]]).astype(np.float32)
        sensor_data = self._current_obs["sensor_data"]
        images = {
            uid: _to_numpy(sensor_data[uid]["rgb"])[0].copy() for uid in CAMERA_UIDS
        }
        self.frames.append(
            {
                "images": images,
                "state": state,
                "action_command_raw": action.copy(),
                "action_command_applied": action.copy(),
                "expert_stage": self.expert_stage,
            }
        )
        transition = self.env.step(action)
        info = transition[4]
        if "lift_height" in info:
            self.max_lift_height = max(
                self.max_lift_height, _scalar_float(info["lift_height"])
            )
        self._current_obs = transition[0]
        return transition


@dataclasses.dataclass(frozen=True)
class AttemptResult:
    seed: int
    accepted: bool
    reason: str
    steps: int
    success_step: int | None
    max_lift_height: float
    shard_path: str | None
    shard_sha256: str | None


def _scalar_bool(value: Any) -> bool:
    return bool(_to_numpy(value).reshape(-1)[0])


def _scalar_float(value: Any) -> float:
    return float(_to_numpy(value).reshape(-1)[0])


def _write_episode(
    output_dir: pathlib.Path,
    seed: int,
    frames: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[pathlib.Path, str]:
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    final_path = episode_dir / f"seed_{seed:010d}.npz"
    if final_path.exists():
        raise FileExistsError(final_path)
    arrays = {
        "observation.images.front": np.stack(
            [frame["images"]["base_camera"] for frame in frames]
        ),
        "observation.images.wrist": np.stack(
            [frame["images"]["wrist_camera"] for frame in frames]
        ),
        "observation.images.side": np.stack(
            [frame["images"]["side_camera"] for frame in frames]
        ),
        "observation.state": np.stack([frame["state"] for frame in frames]),
        "action_command_raw": np.stack(
            [frame["action_command_raw"] for frame in frames]
        ),
        "action_command_applied": np.stack(
            [frame["action_command_applied"] for frame in frames]
        ),
        "expert_stage": np.asarray(
            [frame["expert_stage"] for frame in frames], dtype="U16"
        ),
        "prompt": np.asarray(PROMPT),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    with tempfile.NamedTemporaryFile(
        dir=episode_dir, prefix=f"seed_{seed:010d}.", suffix=".tmp", delete=False
    ) as handle:
        temporary_path = pathlib.Path(handle.name)
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_path, final_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    digest = hashlib.sha256(final_path.read_bytes()).hexdigest()
    return final_path, digest


def collect_attempt(
    *,
    seed: int,
    output_dir: str,
    max_episode_steps: int,
    render_backend: str,
) -> AttemptResult:
    output_path = pathlib.Path(output_dir)
    env = gym.make(
        "LiftCubePiper-v1",
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        sim_backend="physx_cpu",
        num_envs=1,
        max_episode_steps=max_episode_steps,
        render_backend=render_backend,
    )
    recorder = StrictEpisodeRecorder(env)
    try:
        transition = solve(recorder, seed=seed)
        if transition is None:
            return AttemptResult(seed, False, "no_transition", 0, None, 0.0, None, None)
        _, _, terminated, truncated, info = transition
        success = _scalar_bool(info["success"])
        steps = len(recorder.frames)
        max_lift_height = recorder.max_lift_height
        if _scalar_bool(truncated) and not success:
            reason = "timeout"
        elif _scalar_bool(terminated) and not success:
            reason = "terminated_without_success"
        elif not success:
            reason = "task_failure"
        else:
            reason = "accepted"
        if not success:
            return AttemptResult(
                seed, False, reason, steps, None, max_lift_height, None, None
            )
        raw = np.stack([frame["action_command_raw"] for frame in recorder.frames])
        applied = np.stack(
            [frame["action_command_applied"] for frame in recorder.frames]
        )
        if not np.array_equal(raw, applied):
            raise ValueError("Accepted trajectory depends on controller clipping")
        metadata = {
            "seed": seed,
            "max_episode_steps": max_episode_steps,
            "steps": steps,
            "success_step": steps,
            "max_lift_height": max_lift_height,
            "camera_contract": "piper_sim_camera_v1",
            "control_mode": "pd_joint_pos",
            "prompt": PROMPT,
        }
        shard_path, digest = _write_episode(
            output_path, seed, recorder.frames, metadata
        )
        return AttemptResult(
            seed,
            True,
            reason,
            steps,
            steps,
            max_lift_height,
            str(shard_path),
            digest,
        )
    except Exception as error:
        return AttemptResult(
            seed,
            False,
            f"{type(error).__name__}: {error}",
            len(recorder.frames),
            None,
            0.0,
            None,
            None,
        )
    finally:
        recorder.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, required=True)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--num-procs", type=int, default=32)
    parser.add_argument("--render-backends", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_seeds <= 0 or args.num_procs <= 0:
        raise ValueError("num-seeds and num-procs must be positive")
    render_backends = tuple(
        value.strip() for value in args.render_backends.split(",") if value.strip()
    )
    if not render_backends:
        raise ValueError("At least one render backend is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    results: list[AttemptResult] = []
    if args.num_procs == 1:
        for index, seed in enumerate(seeds):
            result = collect_attempt(
                seed=seed,
                output_dir=str(args.output_dir),
                max_episode_steps=args.max_episode_steps,
                render_backend=render_backends[index % len(render_backends)],
            )
            results.append(result)
            print(json.dumps(dataclasses.asdict(result), sort_keys=True), flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_procs, mp_context=context
        ) as executor:
            futures = {
                executor.submit(
                    collect_attempt,
                    seed=seed,
                    output_dir=str(args.output_dir),
                    max_episode_steps=args.max_episode_steps,
                    render_backend=render_backends[index % len(render_backends)],
                ): seed
                for index, seed in enumerate(seeds)
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    json.dumps(dataclasses.asdict(result), sort_keys=True),
                    flush=True,
                )

    results.sort(key=lambda item: item.seed)
    manifest = {
        "attempted": len(results),
        "accepted": sum(result.accepted for result in results),
        "max_episode_steps": args.max_episode_steps,
        "results": [dataclasses.asdict(result) for result in results],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
