#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import multiprocessing
import pathlib

from mani_skill import ASSET_DIR
from mani_skill.envs.tasks.pick_anything.episode_specs import ObjectSpec
from mani_skill.examples.motionplanning.piper.grasping.antipodal import (
    ANTIPODAL_PROVIDER_VERSION,
    AntipodalConfig,
    AntipodalGraspProvider,
)
from mani_skill.examples.motionplanning.piper.grasping.cache import (
    GraspCache,
    cache_key,
    dependency_files_hash,
)
from mani_skill.examples.motionplanning.piper.grasping.contracts import (
    GraspProviderName,
)
from mani_skill.examples.motionplanning.piper.grasping.geometry import (
    GEOMETRY_PREPROCESS_VERSION,
    resolve_object_geometry,
)


def piper_gripper_geometry_hash() -> str:
    root = ASSET_DIR.parent / "robots/piper"
    if not root.is_dir():
        root = pathlib.Path(__file__).parents[1] / "mani_skill/assets/robots/piper"
    paths = [
        root / "piper_description.urdf",
        root / "piper_description.srdf",
        root / "meshes/gripper_base.convex.stl",
        root / "meshes/link7.convex.stl",
        root / "meshes/link8.convex.stl",
    ]
    return dependency_files_hash(paths)


def generate_one(*, object_spec_dict: dict, cache_root: str, config_dict: dict) -> dict:
    spec = ObjectSpec.from_dict(object_spec_dict)
    geometry = resolve_object_geometry(spec)
    config = AntipodalConfig(**config_dict)
    gripper_hash = piper_gripper_geometry_hash()
    key = cache_key(
        geometry,
        provider=GraspProviderName.ANTIPODAL,
        provider_version=ANTIPODAL_PROVIDER_VERSION,
        provider_config_fingerprint=config.fingerprint,
        gripper_geometry_hash=gripper_hash,
    )
    seed = int(key[:16], 16)
    cache = GraspCache(pathlib.Path(cache_root))
    manifest = {
        "object_stable_id": spec.stable_id,
        "canonical_geometry_hash": geometry.canonical_geometry_hash,
        "geometry_preprocess_version": GEOMETRY_PREPROCESS_VERSION,
        "provider": GraspProviderName.ANTIPODAL.value,
        "provider_version": ANTIPODAL_PROVIDER_VERSION,
        "provider_config": dataclasses.asdict(config),
        "provider_config_fingerprint": config.fingerprint,
        "gripper_geometry_hash": gripper_hash,
        "rng_seed": seed,
    }
    try:
        candidates = AntipodalGraspProvider(config).generate(geometry, seed=seed)
    except RuntimeError as error:
        if not str(error).startswith("proposal-empty:"):
            raise
        cache.store(key, [], manifest=manifest, failure=str(error))
        return {"object": spec.stable_id, "key": key, "count": 0, "failure": str(error)}
    cache.store(key, candidates, manifest=manifest)
    return {
        "object": spec.stable_id,
        "key": key,
        "count": len(candidates),
        "failure": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--num-procs", type=int, default=32)
    args = parser.parse_args()
    if args.num_procs <= 0:
        raise ValueError("num-procs must be positive")
    payload = json.loads(args.object_manifest.read_text())
    rows = payload["objects"] if isinstance(payload, dict) else payload
    specs = [ObjectSpec.from_dict(row) for row in rows]
    config = AntipodalConfig()
    context = multiprocessing.get_context("spawn")
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_procs, mp_context=context
    ) as executor:
        futures = [
            executor.submit(
                generate_one,
                object_spec_dict=spec.to_dict(),
                cache_root=str(args.cache_dir),
                config_dict=dataclasses.asdict(config),
            )
            for spec in specs
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    results.sort(key=lambda row: row["object"])
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(
            {
                "provider": GraspProviderName.ANTIPODAL.value,
                "provider_version": ANTIPODAL_PROVIDER_VERSION,
                "provider_config": dataclasses.asdict(config),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
