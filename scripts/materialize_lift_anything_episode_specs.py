#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import pathlib

import gymnasium as gym

import mani_skill.envs  # noqa: F401
from mani_skill.envs.tasks.pick_anything.episode_specs import ObjectSpec


def materialization_coordinates(
    spec: ObjectSpec,
    *,
    local_index: int,
    episode_index_offset: int,
    seed_start: int,
    max_attempts: int,
) -> tuple[str, int]:
    if local_index < 0 or episode_index_offset < 0:
        raise ValueError("Episode indices must be non-negative")
    global_index = episode_index_offset + local_index
    return (
        f"{spec.source}-{global_index:06d}",
        seed_start + global_index * max_attempts,
    )


def _materialize(
    *,
    object_spec_dict: dict,
    stable_episode_id: str,
    environment_seed_start: int,
    max_attempts: int,
    render_backend: str,
) -> dict:
    object_spec = ObjectSpec.from_dict(object_spec_dict)
    env = gym.make(
        "LiftAnythingPiper-v1",
        object_spec=object_spec.to_dict(),
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        sim_backend="physx_cpu",
        num_envs=1,
        render_backend=render_backend,
    )
    attempts = []
    try:
        for attempt_index in range(max_attempts):
            environment_seed = environment_seed_start + attempt_index
            try:
                env.reset(seed=environment_seed, options={"reconfigure": True})
            except RuntimeError as error:
                if "spawn-invalid:" not in str(error):
                    raise
                attempts.append(
                    {
                        "environment_seed": environment_seed,
                        "status": "rejected",
                        "reason": str(error),
                    }
                )
                continue
            attempts.append(
                {
                    "environment_seed": environment_seed,
                    "status": "accepted",
                    "reason": None,
                }
            )
            return {
                "stable_episode_id": stable_episode_id,
                "episode": env.unwrapped.capture_episode_spec(stable_episode_id).to_dict(),
                "attempts": attempts,
            }
    finally:
        env.close()
    return {
        "stable_episode_id": stable_episode_id,
        "episode": None,
        "attempts": attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episode-index-offset", type=int, default=0)
    parser.add_argument("--num-procs", type=int, default=32)
    parser.add_argument("--render-backends", default="cuda:0")
    parser.add_argument("--max-attempts-per-object", type=int, default=8)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.object_manifest.read_text())
    rows = payload["objects"] if isinstance(payload, dict) else payload
    specs = [ObjectSpec.from_dict(row) for row in rows]
    render_backends = tuple(item.strip() for item in args.render_backends.split(",") if item.strip())
    if (
        args.num_procs <= 0
        or not render_backends
        or args.max_attempts_per_object <= 0
        or args.episode_index_offset < 0
    ):
        raise ValueError(
            "num-procs, render-backends, and max-attempts-per-object must be "
            "non-empty, and episode-index-offset must be non-negative"
        )
    context = multiprocessing.get_context("spawn")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_procs, mp_context=context) as executor:
        futures = []
        for index, spec in enumerate(specs):
            stable_episode_id, environment_seed_start = materialization_coordinates(
                spec,
                local_index=index,
                episode_index_offset=args.episode_index_offset,
                seed_start=args.seed_start,
                max_attempts=args.max_attempts_per_object,
            )
            futures.append(
                executor.submit(
                    _materialize,
                    object_spec_dict=spec.to_dict(),
                    stable_episode_id=stable_episode_id,
                    environment_seed_start=environment_seed_start,
                    max_attempts=args.max_attempts_per_object,
                    render_backend=render_backends[index % len(render_backends)],
                )
            )
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["stable_episode_id"])
    episodes = [row["episode"] for row in results if row["episode"] is not None]
    failures = [row for row in results if row["episode"] is None]
    episodes.sort(key=lambda row: row["stable_episode_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "episodes": episodes,
                "materialization": results,
                "attempted_objects": len(results),
                "accepted_objects": len(episodes),
                "failed_objects": len(failures),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if failures and not args.allow_failures:
        failed_ids = [row["stable_episode_id"] for row in failures]
        raise RuntimeError(
            f"Failed to materialize {len(failures)} objects after "
            f"{args.max_attempts_per_object} attempts: {failed_ids}"
        )


if __name__ == "__main__":
    main()
