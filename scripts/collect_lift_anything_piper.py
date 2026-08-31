#!/usr/bin/env python3

from __future__ import annotations

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
from mani_skill.envs.tasks.pick_anything.episode_specs import (
    EpisodeSpec,
    load_episode_specs_manifest,
)
from mani_skill.examples.motionplanning.piper.grasping.contracts import (
    GraspProviderName,
    PipelineName,
)
from mani_skill.examples.motionplanning.piper.solutions.lift_anything import solve

CAMERA_UIDS = ("base_camera", "wrist_camera", "side_camera")
PROMPT = "pick up the object"
MIN_PROJECTED_SIZE = 8.0


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_matrix(value: Any, shape: tuple[int, int]) -> np.ndarray:
    array = _to_numpy(value)
    if array.shape == shape:
        return array
    if array.shape == (1, *shape):
        return array[0]
    raise ValueError(
        f"Expected matrix shape {shape} or {(1, *shape)}, got {array.shape}"
    )


def projected_bbox_size(
    world_points: np.ndarray,
    extrinsic_cv: np.ndarray,
    intrinsic_cv: np.ndarray,
) -> tuple[float, float]:
    points = np.asarray(world_points, dtype=np.float64)
    extrinsic = np.asarray(extrinsic_cv, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_cv, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"world_points must have shape (N, 3), got {points.shape}")
    camera = (
        extrinsic @ np.concatenate([points, np.ones((len(points), 1))], axis=1).T
    ).T
    if np.any(camera[:, 2] <= 0.0):
        raise RuntimeError(
            "spawn-invalid: object OBB crosses or is behind camera plane"
        )
    pixels_h = (intrinsic @ camera.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    size = pixels.max(axis=0) - pixels.min(axis=0)
    return float(size[0]), float(size[1])


def validate_camera_visibility(base_env) -> dict[str, tuple[float, float]]:
    mesh = base_env.obj.get_first_collision_mesh(to_world_frame=False)
    if mesh is None:
        raise RuntimeError("spawn-invalid: object has no collision mesh")
    corners = np.asarray(mesh.bounding_box.vertices, dtype=np.float64)
    actor_matrix = _first_matrix(base_env.obj.pose.to_transformation_matrix(), (4, 4))
    world_points = (actor_matrix[:3, :3] @ corners.T + actor_matrix[:3, 3:4]).T
    params = base_env.get_sensor_params()
    sizes = {}
    for camera_uid in ("base_camera", "side_camera"):
        width, height = projected_bbox_size(
            world_points,
            _first_matrix(params[camera_uid]["extrinsic_cv"], (3, 4)),
            _first_matrix(params[camera_uid]["intrinsic_cv"], (3, 3)),
        )
        if width < MIN_PROJECTED_SIZE or height < MIN_PROJECTED_SIZE:
            raise RuntimeError(
                f"spawn-invalid: {camera_uid} projected OBB is {width:.2f}x{height:.2f} pixels"
            )
        sizes[camera_uid] = (width, height)
    return sizes


class StrictEpisodeRecorder(gym.Wrapper):
    """Record o_t and state_t immediately before the expert submits a_t."""

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
        validate_camera_visibility(self.unwrapped)
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Invalid PIPER action: {action}")
        if not self.action_space.contains(action):
            raise ValueError(f"PIPER expert action would be clipped: {action}")
        if self._current_obs is None:
            raise RuntimeError("Recorder step called before reset")
        qpos = _to_numpy(self.unwrapped.agent.robot.get_qpos())[0]
        sensor_data = self._current_obs["sensor_data"]
        self.frames.append(
            {
                "images": {
                    uid: _to_numpy(sensor_data[uid]["rgb"])[0].copy()
                    for uid in CAMERA_UIDS
                },
                "state": np.concatenate([qpos[:6], qpos[6:7]]).astype(np.float32),
                "action_command_raw": action.copy(),
                "action_command_applied": action.copy(),
                "expert_stage": self.expert_stage,
            }
        )
        transition = self.env.step(action)
        if "lift_height" in transition[4]:
            self.max_lift_height = max(
                self.max_lift_height,
                float(_to_numpy(transition[4]["lift_height"]).reshape(-1)[0]),
            )
        self._current_obs = transition[0]
        return transition


@dataclasses.dataclass(frozen=True)
class AttemptResult:
    stable_episode_id: str
    object_id: str
    source: str
    accepted: bool
    reason: str
    attempted_candidates: int
    steps: int
    max_lift_height: float
    shard_path: str | None
    shard_sha256: str | None
    provider: str = GraspProviderName.OBB.value
    pipeline: str = PipelineName.LEGACY.value
    candidate_evaluations: tuple[dict[str, Any], ...] = ()


def _write_episode(
    output_dir: pathlib.Path,
    spec: EpisodeSpec,
    frames: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[pathlib.Path, str]:
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    final_path = episode_dir / f"{spec.stable_episode_id}.npz"
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
        "episode_spec_json": np.asarray(json.dumps(spec.to_dict(), sort_keys=True)),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    with tempfile.NamedTemporaryFile(
        dir=episode_dir,
        prefix=f"{spec.stable_episode_id}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = pathlib.Path(handle.name)
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_path, final_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return final_path, hashlib.sha256(final_path.read_bytes()).hexdigest()


def collect_attempt(
    *,
    spec_dict: dict[str, Any],
    output_dir: str,
    render_backend: str,
    provider: str = GraspProviderName.OBB.value,
    pipeline: str = PipelineName.LEGACY.value,
    grasp_cache_dir: str | None = None,
) -> AttemptResult:
    spec = EpisodeSpec.from_dict(spec_dict)
    provider_name = GraspProviderName(provider)
    pipeline_name = PipelineName(pipeline)
    if pipeline_name is PipelineName.COMMON and spec.settled_object_state is None:
        raise ValueError("common pipeline requires a post-settle EpisodeSpec")
    output_path = pathlib.Path(output_dir)
    env = gym.make(
        "LiftAnythingPiper-v1",
        episode_spec=spec.to_dict(),
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        sim_backend="physx_cpu",
        num_envs=1,
        max_episode_steps=160 if pipeline_name is PipelineName.COMMON else 100,
        render_backend=render_backend,
        success_streak_steps=10 if pipeline_name is PipelineName.COMMON else 3,
    )
    recorder = StrictEpisodeRecorder(env)
    try:
        result = solve(
            recorder,
            seed=spec.environment_seed,
            provider=provider_name,
            pipeline=pipeline_name,
            grasp_cache_dir=(
                None if grasp_cache_dir is None else pathlib.Path(grasp_cache_dir)
            ),
        )
        evaluations = tuple(item.to_dict() for item in result.evaluations)
        if not result.success:
            return AttemptResult(
                spec.stable_episode_id,
                spec.object_spec.stable_id,
                spec.object_spec.source,
                False,
                result.reason,
                result.attempted_candidates,
                len(recorder.frames),
                recorder.max_lift_height,
                None,
                None,
                provider_name.value,
                pipeline_name.value,
                evaluations,
            )
        raw = np.stack([frame["action_command_raw"] for frame in recorder.frames])
        applied = np.stack(
            [frame["action_command_applied"] for frame in recorder.frames]
        )
        if not np.array_equal(raw, applied):
            raise ValueError("Accepted trajectory depends on controller clipping")
        metadata = {
            "stable_episode_id": spec.stable_episode_id,
            "episode_spec_fingerprint": spec.fingerprint,
            "candidate_id": result.candidate_id,
            "attempted_candidates": result.attempted_candidates,
            "steps": len(recorder.frames),
            "max_lift_height": recorder.max_lift_height,
            "prompt": PROMPT,
            "camera_contract": "piper_sim_camera_v1",
            "control_mode": "pd_joint_pos",
            "provider": provider_name.value,
            "pipeline": pipeline_name.value,
            "candidate_evaluations": evaluations,
        }
        shard_path, digest = _write_episode(
            output_path, spec, recorder.frames, metadata
        )
        return AttemptResult(
            spec.stable_episode_id,
            spec.object_spec.stable_id,
            spec.object_spec.source,
            True,
            "accepted",
            result.attempted_candidates,
            len(recorder.frames),
            recorder.max_lift_height,
            str(shard_path),
            digest,
            provider_name.value,
            pipeline_name.value,
            evaluations,
        )
    except Exception as error:
        return AttemptResult(
            spec.stable_episode_id,
            spec.object_spec.stable_id,
            spec.object_spec.source,
            False,
            f"{type(error).__name__}: {error}",
            0,
            len(recorder.frames),
            recorder.max_lift_height,
            None,
            None,
            provider_name.value,
            pipeline_name.value,
            (),
        )
    finally:
        recorder.close()


def load_specs(
    path: pathlib.Path, *, require_settled_state: bool = False
) -> list[EpisodeSpec]:
    return load_episode_specs_manifest(
        path, require_settled_state=require_settled_state
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--num-procs", type=int, default=32)
    parser.add_argument("--render-backends", default="cuda:0")
    parser.add_argument(
        "--provider",
        choices=[item.value for item in GraspProviderName],
        default=GraspProviderName.OBB.value,
    )
    parser.add_argument(
        "--pipeline",
        choices=[item.value for item in PipelineName],
        default=PipelineName.LEGACY.value,
    )
    parser.add_argument("--grasp-cache-dir", type=pathlib.Path)
    parser.add_argument(
        "--require-settled-state",
        action="store_true",
        help="Reject legacy manifests that cannot guarantee paired post-settle replay.",
    )
    args = parser.parse_args()
    if args.num_procs <= 0:
        raise ValueError("num-procs must be positive")
    specs = load_specs(
        args.episode_manifest,
        require_settled_state=(
            args.require_settled_state or args.pipeline == PipelineName.COMMON.value
        ),
    )
    render_backends = tuple(
        item.strip() for item in args.render_backends.split(",") if item.strip()
    )
    if not render_backends:
        raise ValueError("At least one render backend is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[AttemptResult] = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_procs, mp_context=context
    ) as executor:
        futures = {
            executor.submit(
                collect_attempt,
                spec_dict=spec.to_dict(),
                output_dir=str(args.output_dir),
                render_backend=render_backends[index % len(render_backends)],
                provider=args.provider,
                pipeline=args.pipeline,
                grasp_cache_dir=(
                    None if args.grasp_cache_dir is None else str(args.grasp_cache_dir)
                ),
            ): spec.stable_episode_id
            for index, spec in enumerate(specs)
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(dataclasses.asdict(result), sort_keys=True), flush=True)
    results.sort(key=lambda item: item.stable_episode_id)
    manifest = {
        "episode_manifest": str(args.episode_manifest),
        "episode_manifest_sha256": hashlib.sha256(
            args.episode_manifest.read_bytes()
        ).hexdigest(),
        "attempted": len(results),
        "accepted": sum(result.accepted for result in results),
        "provider": args.provider,
        "pipeline": args.pipeline,
        "results": [dataclasses.asdict(result) for result in results],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
