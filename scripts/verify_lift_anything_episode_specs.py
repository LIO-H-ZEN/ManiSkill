#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import pathlib

import gymnasium as gym
import numpy as np

import mani_skill.envs  # noqa: F401
from mani_skill.envs.tasks.pick_anything.episode_specs import EpisodeSpec


CAMERA_UIDS = ("base_camera", "wrist_camera", "side_camera")


def _verify(spec_dict: dict, render_backend: str) -> dict:
    spec = EpisodeSpec.from_dict(spec_dict)
    env = gym.make(
        "LiftAnythingPiper-v1",
        episode_spec=spec.to_dict(),
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        sim_backend="physx_cpu",
        num_envs=1,
        render_backend=render_backend,
    )
    try:
        obs, _ = env.reset(
            seed=spec.environment_seed,
            options={"reconfigure": True},
        )
        replayed = env.unwrapped.capture_episode_spec(spec.stable_episode_id)
        if replayed.fingerprint != spec.fingerprint:
            raise RuntimeError(
                f"Replay fingerprint mismatch for {spec.stable_episode_id}: "
                f"expected {spec.fingerprint}, got {replayed.fingerprint}"
            )
        shapes = {}
        for camera_uid in CAMERA_UIDS:
            image = obs["sensor_data"][camera_uid]["rgb"]
            if hasattr(image, "detach"):
                image = image.detach().cpu().numpy()
            image = np.asarray(image)
            if image.shape != (1, 224, 224, 3) or image.dtype != np.uint8:
                raise RuntimeError(
                    f"{spec.stable_episode_id} {camera_uid} has invalid image "
                    f"contract: shape={image.shape}, dtype={image.dtype}"
                )
            shapes[camera_uid] = list(image.shape)
        return {
            "stable_episode_id": spec.stable_episode_id,
            "fingerprint": spec.fingerprint,
            "environment_seed": spec.environment_seed,
            "camera_shapes": shapes,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--num-procs", type=int, default=32)
    parser.add_argument("--render-backends", default="cuda:0")
    args = parser.parse_args()
    payload = json.loads(args.episode_manifest.read_text())
    rows = payload["episodes"] if isinstance(payload, dict) else payload
    specs = [EpisodeSpec.from_dict(row) for row in rows]
    render_backends = tuple(
        item.strip() for item in args.render_backends.split(",") if item.strip()
    )
    if args.num_procs <= 0 or not render_backends:
        raise ValueError("num-procs and render-backends must be non-empty")
    context = multiprocessing.get_context("spawn")
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_procs,
        mp_context=context,
    ) as executor:
        futures = [
            executor.submit(
                _verify,
                spec.to_dict(),
                render_backends[index % len(render_backends)],
            )
            for index, spec in enumerate(specs)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["stable_episode_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "episode_manifest": str(args.episode_manifest),
                "verified": len(results),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
