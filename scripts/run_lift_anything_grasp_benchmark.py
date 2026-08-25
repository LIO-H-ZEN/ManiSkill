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
import time

from mani_skill.envs.tasks.pick_anything.episode_specs import EpisodeSpec
from mani_skill.examples.motionplanning.piper.grasping.benchmark import (
    analyze_benchmark,
)
from mani_skill.examples.motionplanning.piper.grasping.contracts import BenchmarkGroup
from scripts.collect_lift_anything_piper import collect_attempt
from scripts.freeze_lift_anything_benchmark import current_frozen_versions


def _write_json_atomic(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=path.name, delete=False
        ) as handle:
            temporary_path = pathlib.Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_completed_result(
    result: dict, episode: EpisodeSpec, group: BenchmarkGroup
) -> None:
    expected = {
        "stable_episode_id": episode.stable_episode_id,
        "object_id": episode.object_spec.stable_id,
        "benchmark_group": group.value,
        "provider": group.provider.value,
        "pipeline": group.pipeline.value,
        "episode_spec_fingerprint": episode.fingerprint,
    }
    actual = {key: result.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"stale or mismatched benchmark shard for {episode.stable_episode_id}: "
            f"expected {expected}, got {actual}"
        )


def _load_frozen_manifest(path: pathlib.Path) -> tuple[dict, list[EpisodeSpec]]:
    payload = json.loads(path.read_text())
    expected = payload.pop("benchmark_manifest_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode()).hexdigest()
    if actual != expected:
        raise ValueError("benchmark manifest SHA-256 mismatch")
    if payload.get("frozen_versions") != current_frozen_versions():
        raise ValueError("benchmark manifest versions do not match this checkout")
    payload["benchmark_manifest_sha256"] = expected
    episodes = [EpisodeSpec.from_dict(row) for row in payload["episodes"]]
    if any(episode.settled_object_state is None for episode in episodes):
        raise ValueError("benchmark manifest contains an episode without settled state")
    episode_ids = [episode.stable_episode_id for episode in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("benchmark manifest contains duplicate episode IDs")
    return payload, episodes


def _run_group(
    group: BenchmarkGroup,
    episodes: list[EpisodeSpec],
    *,
    output_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    render_backends: tuple[str, ...],
    num_procs: int,
) -> list[dict]:
    group_dir = output_dir / group.value
    group_dir.mkdir(parents=True, exist_ok=True)
    result_dir = group_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    results = []

    def completed_result(episode: EpisodeSpec) -> dict | None:
        path = result_dir / f"{episode.stable_episode_id}.json"
        if not path.is_file():
            return None
        result = json.loads(path.read_text())
        _validate_completed_result(result, episode, group)
        return result

    pending = []
    for index, episode in enumerate(episodes):
        existing = completed_result(episode)
        if existing is not None:
            results.append(existing)
        else:
            pending.append((index, episode))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_procs, mp_context=context
    ) as executor:
        futures = {
            executor.submit(
                _run_one,
                group=group.value,
                episode_dict=episode.to_dict(),
                group_dir=str(group_dir),
                result_path=str(result_dir / f"{episode.stable_episode_id}.json"),
                render_backend=render_backends[index % len(render_backends)],
                cache_dir=str(cache_dir),
            ): episode.stable_episode_id
            for index, episode in pending
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps({"group": group.value, **result}, sort_keys=True), flush=True
            )
    return sorted(results, key=lambda row: row["stable_episode_id"])


def _run_one(
    *,
    group: str,
    episode_dict: dict,
    group_dir: str,
    result_path: str,
    render_backend: str,
    cache_dir: str,
) -> dict:
    benchmark_group = BenchmarkGroup(group)
    started_at = time.monotonic()
    result = dataclasses.asdict(
        collect_attempt(
            spec_dict=episode_dict,
            output_dir=group_dir,
            render_backend=render_backend,
            provider=benchmark_group.provider.value,
            pipeline=benchmark_group.pipeline.value,
            grasp_cache_dir=cache_dir,
            record_trajectory=False,
        )
    )
    episode = EpisodeSpec.from_dict(episode_dict)
    result["benchmark_group"] = benchmark_group.value
    result["episode_spec_fingerprint"] = episode.fingerprint
    result["elapsed_seconds"] = time.monotonic() - started_at
    result = json.loads(json.dumps(result))
    final_path = pathlib.Path(result_path)
    _validate_completed_result(result, episode, benchmark_group)
    _write_json_atomic(final_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--grasp-cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--groups", default="L0,A,B")
    parser.add_argument("--num-procs", type=int, default=32)
    parser.add_argument("--render-backends", default="cuda:0")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    if args.num_procs <= 0:
        raise ValueError("num-procs must be positive")
    manifest, episodes = _load_frozen_manifest(args.benchmark_manifest)
    groups = tuple(BenchmarkGroup(value.strip()) for value in args.groups.split(","))
    render_backends = tuple(
        value.strip() for value in args.render_backends.split(",") if value.strip()
    )
    if not groups or not render_backends:
        raise ValueError("groups and render-backends must not be empty")
    if len(groups) != len(set(groups)):
        raise ValueError("groups must not contain duplicates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_rows = {}
    group_wall_seconds = {}
    for group in groups:
        started_at = time.monotonic()
        group_rows[group.value] = _run_group(
            group,
            episodes,
            output_dir=args.output_dir,
            cache_dir=args.grasp_cache_dir,
            render_backends=render_backends,
            num_procs=args.num_procs,
        )
        group_wall_seconds[group.value] = time.monotonic() - started_at
    episode_to_object = {
        episode.stable_episode_id: episode.object_spec.stable_id for episode in episodes
    }
    report = {
        "benchmark_manifest_sha256": manifest["benchmark_manifest_sha256"],
        "benchmark_split": manifest["split"],
        "group_wall_seconds": group_wall_seconds,
        "results": group_rows,
        "summary": analyze_benchmark(
            group_rows,
            episode_to_object,
            manifest["object_categories"],
            bootstrap_samples=args.bootstrap_samples,
            seed=int(manifest["seed"]),
            evaluate_promotion=bool(manifest.get("promotion_eligible", True)),
        ),
    }
    _write_json_atomic(args.output_dir / "benchmark_results.json", report)


if __name__ == "__main__":
    main()
