#!/usr/bin/env python3

"""Convert offline GraspGen-X YAML into the shared PiPER grasp cache."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from mani_skill.envs.tasks.pick_anything.general_pickup_specs import (
    RoboDojoLayoutSpec,
)
from mani_skill.examples.motionplanning.piper.grasping.cache import (
    GraspCache,
    cache_key,
    piper_gripper_geometry_hash,
)
from mani_skill.examples.motionplanning.piper.grasping.contracts import (
    GraspProviderName,
)
from mani_skill.examples.motionplanning.piper.grasping.geometry import (
    GEOMETRY_PREPROCESS_VERSION,
    resolve_object_geometry,
)
from mani_skill.examples.motionplanning.piper.grasping.graspgenx import (
    GRASPGENX_PROVIDER_VERSION,
    GraspGenXConfig,
    GraspGenXGraspProvider,
)
from mani_skill.examples.motionplanning.piper.grasping.gripper_geometry import (
    PiperGripperGeometry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-contract", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--grasp-yaml", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()

    layout = RoboDojoLayoutSpec.load(args.layout_contract)
    geometry = resolve_object_geometry(
        layout.target.object_spec, asset_root=args.asset_root
    )
    config = GraspGenXConfig()
    provider = GraspGenXGraspProvider(config)
    generation_manifest = provider.validate_generation_manifest(args.grasp_yaml)
    raw_candidates = provider.load_isaac_yaml(args.grasp_yaml, geometry)
    candidates = provider.refine_contact_candidates(
        raw_candidates,
        geometry,
        PiperGripperGeometry.from_package_assets(),
    )
    gripper_hash = piper_gripper_geometry_hash()
    fingerprint = provider.cache_fingerprint(geometry)
    key = cache_key(
        geometry,
        provider=GraspProviderName.GRASPGENX,
        provider_version=GRASPGENX_PROVIDER_VERSION,
        provider_config_fingerprint=fingerprint,
        gripper_geometry_hash=gripper_hash,
    )
    GraspCache(args.cache_dir).store(
        key,
        candidates,
        manifest={
            "object_stable_id": layout.target.object_spec.stable_id,
            "canonical_geometry_hash": geometry.canonical_geometry_hash,
            "geometry_preprocess_version": GEOMETRY_PREPROCESS_VERSION,
            "provider": GraspProviderName.GRASPGENX.value,
            "provider_version": GRASPGENX_PROVIDER_VERSION,
            "provider_config": dataclasses.asdict(config),
            "provider_config_fingerprint": fingerprint,
            "gripper_geometry_hash": gripper_hash,
            "source_yaml": str(args.grasp_yaml.resolve()),
            "source_yaml_sha256": hashlib.sha256(
                args.grasp_yaml.read_bytes()
            ).hexdigest(),
            "generation_manifest": generation_manifest,
            "raw_candidate_count": len(raw_candidates),
            "refined_candidate_count": len(candidates),
        },
    )
    print(
        json.dumps(
            {
                "object": layout.target.object_spec.stable_id,
                "key": key,
                "candidate_count": len(candidates),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
