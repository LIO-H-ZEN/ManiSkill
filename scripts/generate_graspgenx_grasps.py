#!/usr/bin/env python3

"""Run pinned, deterministic GraspGen-X complete-mesh inference.

This script is intentionally executed with the isolated GraspGen-X uv
environment. It writes the upstream Isaac-grasp YAML plus a generation
manifest that the ManiSkill importer validates before caching proposals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import runpy
import subprocess
import sys
from pathlib import Path

IMPLEMENTATION_REVISION = "b9429097728cb1c430dd78b92edf17ba318aad03"
CHECKPOINT_REVISION = "7c834043c11a11417e31d6d5ea9355801e40a2c1"
GRIPPER_ASSETS_REVISION = "19a03c00d19aeaf052d0f6801f0041982d676e8a"
GRIPPER_NAME = "piper_hand"
CONTRACT_VERSION = "graspgenx_complete_mesh_generation_v1"
GENERATION_SEED = 0
NUM_GRASPS = 1024
TOPK_NUM_GRASPS = 256
NUM_SAMPLE_POINTS = 3500
GRASP_THRESHOLD = -1.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _seed_everything() -> None:
    os.environ["PYTHONHASHSEED"] = str(GENERATION_SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch

    random.seed(GENERATION_SEED)
    np.random.seed(GENERATION_SEED)
    torch.manual_seed(GENERATION_SEED)
    torch.cuda.manual_seed_all(GENERATION_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graspgenx-root", type=Path, required=True)
    parser.add_argument("--mesh-file", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()

    root = args.graspgenx_root.resolve()
    demo = root / "scripts/demo_object_mesh.py"
    for path in (demo, args.mesh_file, args.checkpoints, args.assets_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if _git_revision(root) != IMPLEMENTATION_REVISION:
        raise RuntimeError("GraspGen-X source revision does not match the contract")
    output = args.output_file.resolve()
    manifest_path = output.with_suffix(output.suffix + ".generation.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(output if output.exists() else manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    _seed_everything()
    scripts_dir = str(demo.parent)
    sys.path[:0] = [scripts_dir, str(root)]
    sys.argv = [
        str(demo),
        "--mesh_file",
        str(args.mesh_file.resolve()),
        "--mesh_scale",
        "1.0",
        "--checkpoints",
        str(args.checkpoints.resolve()),
        "--gripper_name",
        GRIPPER_NAME,
        "--assets_dir",
        str(args.assets_dir.resolve()),
        "--grasp_threshold",
        str(GRASP_THRESHOLD),
        "--num_grasps",
        str(NUM_GRASPS),
        "--return_topk",
        "--topk_num_grasps",
        str(TOPK_NUM_GRASPS),
        "--num_sample_points",
        str(NUM_SAMPLE_POINTS),
        "--output_file",
        str(output),
        "--no-visualization",
    ]
    runpy.run_path(str(demo), run_name="__main__")
    if not output.is_file():
        raise RuntimeError("GraspGen-X completed without writing its YAML output")

    import yaml

    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    rows = payload.get("grasps") if isinstance(payload, dict) else None
    if not isinstance(rows, dict) or len(rows) != TOPK_NUM_GRASPS:
        raise RuntimeError(
            "GraspGen-X output violates the fixed top-k contract: "
            f"expected {TOPK_NUM_GRASPS}, got "
            f"{len(rows) if isinstance(rows, dict) else 'invalid'}"
        )
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "gripper_assets_revision": GRIPPER_ASSETS_REVISION,
        "gripper_name": GRIPPER_NAME,
        "seed": GENERATION_SEED,
        "num_grasps": NUM_GRASPS,
        "topk_num_grasps": TOPK_NUM_GRASPS,
        "num_sample_points": NUM_SAMPLE_POINTS,
        "grasp_threshold": GRASP_THRESHOLD,
        "mesh_file": str(args.mesh_file.resolve()),
        "mesh_sha256": _sha256(args.mesh_file),
        "output_sha256": _sha256(output),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
