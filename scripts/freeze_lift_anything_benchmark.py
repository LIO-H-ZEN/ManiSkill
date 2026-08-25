#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.envs.tasks.pick_anything.episode_specs import (
    EpisodeSpec,
    load_episode_specs_manifest,
)
from mani_skill.examples.motionplanning.piper.grasping.antipodal import (
    ANTIPODAL_PROVIDER_VERSION,
    AntipodalConfig,
)
from mani_skill.examples.motionplanning.piper.grasping.contracts import (
    FRAME_CONVENTION_VERSION,
)
from mani_skill.examples.motionplanning.piper.grasping.geometry import (
    GEOMETRY_PREPROCESS_VERSION,
)
from mani_skill.examples.motionplanning.piper.grasping.pipeline import (
    COMMON_PIPELINE_VERSION,
)
from mani_skill.examples.motionplanning.piper.motionplanner import (
    PiperMotionPlanningSolver,
)

BENCHMARK_CATEGORIES = (
    "simple_convex",
    "thin_flat",
    "ring_u_concave",
    "multipart_slender",
)
QUICK_CATEGORY = "unstratified"


def current_frozen_versions() -> dict[str, str]:
    piper_urdf = pathlib.Path(PACKAGE_ASSET_DIR) / "robots/piper/piper_description.urdf"
    return {
        "frame_convention": FRAME_CONVENTION_VERSION,
        "geometry_preprocess": GEOMETRY_PREPROCESS_VERSION,
        "antipodal_provider": ANTIPODAL_PROVIDER_VERSION,
        "antipodal_config_fingerprint": AntipodalConfig().fingerprint,
        "common_pipeline": COMMON_PIPELINE_VERSION,
        "collision_proxy_version": PiperMotionPlanningSolver.COLLISION_PROXY_VERSION,
        "collision_model_fingerprint": (
            PiperMotionPlanningSolver.collision_model_fingerprint(str(piper_urdf))
        ),
    }


def freeze_manifest(
    episodes: list[EpisodeSpec],
    object_categories: dict[str, str],
    *,
    split: str,
    objects_per_category: int,
    poses_per_object: int,
    seed: int,
    excluded_objects: set[str] | None = None,
) -> dict:
    if split not in ("pilot", "formal"):
        raise ValueError("split must be pilot or formal")
    if objects_per_category <= 0 or poses_per_object <= 0:
        raise ValueError("object and pose counts must be positive")
    excluded_objects = excluded_objects or set()
    invalid_categories = sorted(
        set(object_categories.values()) - set(BENCHMARK_CATEGORIES)
    )
    if invalid_categories:
        raise ValueError(f"Unknown benchmark categories: {invalid_categories}")
    by_object: dict[str, list[EpisodeSpec]] = {}
    episode_ids = set()
    for episode in episodes:
        if episode.stable_episode_id in episode_ids:
            raise ValueError(f"Duplicate episode ID: {episode.stable_episode_id}")
        episode_ids.add(episode.stable_episode_id)
        if episode.settled_object_state is None:
            raise ValueError(
                f"Episode {episode.stable_episode_id} has no post-settle state"
            )
        by_object.setdefault(episode.object_spec.stable_id, []).append(episode)
    rng = np.random.default_rng(seed)
    selected_objects = []
    for category in BENCHMARK_CATEGORIES:
        choices = sorted(
            object_id
            for object_id, assigned in object_categories.items()
            if assigned == category
            and object_id not in excluded_objects
            and len(by_object.get(object_id, ())) >= poses_per_object
        )
        if len(choices) < objects_per_category:
            raise ValueError(
                f"Category {category} has {len(choices)} eligible objects, "
                f"requires {objects_per_category}"
            )
        order = rng.permutation(len(choices))[:objects_per_category]
        selected_objects.extend(choices[int(index)] for index in order)
    selected_episodes = []
    for object_id in sorted(selected_objects):
        object_episodes = sorted(
            by_object[object_id], key=lambda item: item.stable_episode_id
        )
        selected_episodes.extend(object_episodes[:poses_per_object])
    payload = {
        "schema_version": "lift_anything_grasp_benchmark_v1",
        "split": split,
        "seed": seed,
        "objects_per_category": objects_per_category,
        "poses_per_object": poses_per_object,
        "categories": list(BENCHMARK_CATEGORIES),
        "object_categories": {
            object_id: object_categories[object_id]
            for object_id in sorted(selected_objects)
        },
        "frozen_versions": current_frozen_versions(),
        "episodes": [episode.to_dict() for episode in selected_episodes],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["benchmark_manifest_sha256"] = hashlib.sha256(
        canonical.encode()
    ).hexdigest()
    return payload


def freeze_quick_manifest(
    episodes: list[EpisodeSpec], *, episode_count: int, seed: int
) -> dict:
    if episode_count <= 0:
        raise ValueError("episode_count must be positive")
    if len(episodes) != episode_count:
        raise ValueError(
            f"Quick benchmark requires exactly {episode_count} episodes, "
            f"got {len(episodes)}"
        )
    episode_ids = [episode.stable_episode_id for episode in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("Quick benchmark contains duplicate episode IDs")
    object_ids = [episode.object_spec.stable_id for episode in episodes]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("Quick benchmark requires one layout per distinct object")
    missing_settled = [
        episode.stable_episode_id
        for episode in episodes
        if episode.settled_object_state is None
    ]
    if missing_settled:
        raise ValueError(
            f"Quick benchmark episodes lack post-settle state: {missing_settled}"
        )
    ordered = sorted(episodes, key=lambda episode: episode.stable_episode_id)
    payload = {
        "schema_version": "lift_anything_grasp_benchmark_v1",
        "split": "quick",
        "selection_mode": "unstratified_distinct_objects",
        "promotion_eligible": False,
        "seed": seed,
        "episode_count": episode_count,
        "poses_per_object": 1,
        "categories": [QUICK_CATEGORY],
        "object_categories": {
            episode.object_spec.stable_id: QUICK_CATEGORY for episode in ordered
        },
        "frozen_versions": current_frozen_versions(),
        "episodes": [episode.to_dict() for episode in ordered],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["benchmark_manifest_sha256"] = hashlib.sha256(
        canonical.encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--category-map", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--split", choices=("quick", "pilot", "formal"), required=True)
    parser.add_argument("--objects-per-category", type=int)
    parser.add_argument("--poses-per-object", type=int)
    parser.add_argument("--episode-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--exclude-manifest", type=pathlib.Path)
    args = parser.parse_args()
    episodes = load_episode_specs_manifest(
        args.episode_manifest, require_settled_state=True
    )
    if args.split == "quick":
        if any(
            value is not None
            for value in (
                args.category_map,
                args.objects_per_category,
                args.poses_per_object,
                args.exclude_manifest,
            )
        ):
            raise ValueError(
                "quick split does not accept category or stratified-selection options"
            )
        payload = freeze_quick_manifest(
            episodes, episode_count=args.episode_count, seed=args.seed
        )
    else:
        if args.category_map is None:
            raise ValueError("pilot/formal splits require --category-map")
        objects_per_category = args.objects_per_category or (
            6 if args.split == "pilot" else 25
        )
        poses_per_object = args.poses_per_object or 5
        categories = json.loads(args.category_map.read_text())
        excluded = set()
        if args.exclude_manifest is not None:
            previous = json.loads(args.exclude_manifest.read_text())
            excluded = set(previous["object_categories"])
        payload = freeze_manifest(
            episodes,
            categories,
            split=args.split,
            objects_per_category=objects_per_category,
            poses_per_object=poses_per_object,
            seed=args.seed,
            excluded_objects=excluded,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
