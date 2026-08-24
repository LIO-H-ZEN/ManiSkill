#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib

from mani_skill.envs.tasks.pick_anything.episode_specs import EpisodeSpec, ObjectSpec


def build_supplemental_specs(
    episodes: list[EpisodeSpec],
    expert_results: list[dict],
    *,
    source_counts: dict[str, int],
    source_status: dict[str, str] | None = None,
    unique_sources: set[str] | None = None,
) -> tuple[list[ObjectSpec], dict[str, dict[str, int]]]:
    source_status = source_status or dict.fromkeys(source_counts, "failed")
    unique_sources = unique_sources or {"interndata"}
    invalid_status = set(source_status.values()) - {"failed", "accepted", "all"}
    if invalid_status:
        raise ValueError(f"Invalid expert selection status: {sorted(invalid_status)}")
    by_episode_id = {episode.stable_episode_id: episode for episode in episodes}
    if len(by_episode_id) != len(episodes):
        raise ValueError("Episode manifests contain duplicate stable IDs")

    candidates_by_source: dict[str, list[ObjectSpec]] = {
        source: [] for source in source_counts
    }
    seen_result_ids: set[str] = set()
    for result in expert_results:
        episode_id = result.get("stable_episode_id")
        if not isinstance(episode_id, str) or episode_id in seen_result_ids:
            raise ValueError(f"Invalid or duplicate expert result ID: {episode_id!r}")
        seen_result_ids.add(episode_id)
        episode = by_episode_id.get(episode_id)
        if episode is None:
            raise ValueError(f"Expert result has no EpisodeSpec: {episode_id}")
        source = episode.object_spec.source
        if source not in candidates_by_source:
            continue
        status = source_status.get(source, "failed")
        accepted = bool(result.get("accepted"))
        if status == "all" or accepted == (status == "accepted"):
            candidates_by_source[source].append(episode.object_spec)

    selected: list[ObjectSpec] = []
    metadata: dict[str, dict[str, int]] = {}
    for source, count in source_counts.items():
        if count < 0:
            raise ValueError(f"Supplement count must be non-negative: {source}={count}")
        pool = candidates_by_source[source]
        if source in unique_sources:
            unique_pool = []
            seen_objects: set[str] = set()
            for spec in pool:
                if spec.stable_id not in seen_objects:
                    unique_pool.append(spec)
                    seen_objects.add(spec.stable_id)
            pool = unique_pool
        if count and not pool:
            raise RuntimeError(f"No failed {source} EpisodeSpecs available")
        chosen = [pool[index % len(pool)] for index in range(count)] if count else []
        selected.extend(chosen)
        metadata[source] = {
            "requested": count,
            "status": source_status.get(source, "failed"),
            "matching_pool": len(candidates_by_source[source]),
            "selection_pool": len(pool),
            "unique_selected_objects": len({spec.stable_id for spec in chosen}),
        }
    return selected, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-manifests", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--expert-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--cube-count", type=int, default=16)
    parser.add_argument("--ycb-count", type=int, default=300)
    parser.add_argument("--interndata-count", type=int, default=320)
    parser.add_argument("--cube-status", choices=("failed", "accepted", "all"), default="failed")
    parser.add_argument("--ycb-status", choices=("failed", "accepted", "all"), default="failed")
    parser.add_argument(
        "--interndata-status",
        choices=("failed", "accepted", "all"),
        default="failed",
    )
    parser.add_argument("--unique-sources", default="interndata")
    args = parser.parse_args()

    episodes = []
    for path in args.episode_manifests:
        payload = json.loads(path.read_text())
        episodes.extend(EpisodeSpec.from_dict(row) for row in payload["episodes"])
    expert_payload = json.loads(args.expert_manifest.read_text())
    results = expert_payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Expert manifest is missing results")

    specs, metadata = build_supplemental_specs(
        episodes,
        results,
        source_counts={
            "cube": args.cube_count,
            "ycb": args.ycb_count,
            "interndata": args.interndata_count,
        },
        source_status={
            "cube": args.cube_status,
            "ycb": args.ycb_status,
            "interndata": args.interndata_status,
        },
        unique_sources={
            source.strip()
            for source in args.unique_sources.split(",")
            if source.strip()
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "objects": [spec.to_dict() for spec in specs],
                "selection": metadata,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
