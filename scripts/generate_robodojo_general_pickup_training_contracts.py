#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mani_skill.envs.tasks.pick_anything.general_pickup_training_contracts import (
    generate_training_contracts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-contract-root", type=Path, required=True)
    parser.add_argument("--converted-asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--first-layout-id", type=int, default=1000)
    parser.add_argument("--placement-margin", type=float, default=0.02)
    parser.add_argument("--target-x-min", type=float, default=-0.25)
    parser.add_argument("--target-x-max", type=float, default=0.25)
    parser.add_argument("--target-y-min", type=float, default=-0.25)
    parser.add_argument("--target-y-max", type=float, default=0.0)
    args = parser.parse_args()
    manifest = generate_training_contracts(
        source_contract_root=args.source_contract_root.resolve(),
        converted_asset_root=args.converted_asset_root.resolve(),
        output_root=args.output_root.resolve(),
        count=args.count,
        seed=args.seed,
        first_layout_id=args.first_layout_id,
        margin=args.placement_margin,
        target_xlim=(args.target_x_min, args.target_x_max),
        target_ylim=(args.target_y_min, args.target_y_max),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
