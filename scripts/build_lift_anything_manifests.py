#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import pathlib
import time
from typing import Any

import numpy as np
import trimesh

from mani_skill import ASSET_DIR
from mani_skill.envs.tasks.pick_anything.episode_specs import ObjectSpec
from mani_skill.envs.tasks.pick_anything.randomization.object_sources import (
    InternDataAssetsSource,
    YCBSource,
)


SPLIT_SEED = 42


def _rank(value: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{value}".encode()).hexdigest()


def _split(values: list[str], counts: tuple[int, int, int]) -> dict[str, list[str]]:
    ordered = sorted(values, key=lambda item: (_rank(item), item))
    if sum(counts) != len(ordered):
        raise ValueError(f"Split counts {counts} do not cover {len(ordered)} values")
    train, validation, formal = counts
    return {
        "train": ordered[:train],
        "validation": ordered[train : train + validation],
        "formal": ordered[train + validation :],
    }


@dataclasses.dataclass(frozen=True)
class AuditRow:
    stable_id: str
    source: str
    category: str | None
    object_id: str
    local_path: str | None
    status: str
    extents_m: tuple[float, float, float] | None
    detail: str | None


def _audit_interndata(item: tuple[str, str]) -> AuditRow:
    category, object_id = item
    root = (
        InternDataAssetsSource.CACHE_DIR
        / InternDataAssetsSource.REPO_PREFIX
        / category
        / object_id
    )
    path = root / "Aligned.obj"
    stable_id = f"interndata/{category}/{object_id}"
    if not path.is_file():
        return AuditRow(stable_id, "interndata", category, object_id, None, "not_cached", None, None)
    try:
        mesh = trimesh.load_mesh(path, process=False)
        extents = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64) * 0.001
        if extents.shape != (3,) or not np.all(np.isfinite(extents)) or np.any(extents <= 0):
            raise ValueError(f"invalid bounds/extents {extents}")
        return AuditRow(
            stable_id,
            "interndata",
            category,
            object_id,
            str(path),
            "loadable",
            tuple(float(item) for item in extents),
            None,
        )
    except Exception as error:
        return AuditRow(
            stable_id,
            "interndata",
            category,
            object_id,
            str(path),
            "invalid_mesh",
            None,
            f"{type(error).__name__}: {error}",
        )


def _cube_specs(count: int, prefix: str) -> list[ObjectSpec]:
    rng = np.random.RandomState(SPLIT_SEED + sum(prefix.encode("utf-8")))
    return [
        ObjectSpec(
            "cube",
            f"{prefix}-{index:04d}",
            cube_half_size=float(rng.uniform(0.015, 0.03)),
            cube_color=tuple([*rng.uniform(0.1, 1.0, size=3).tolist(), 1.0]),
        )
        for index in range(count)
    ]


def _repeat_specs(specs: list[ObjectSpec], count: int) -> list[ObjectSpec]:
    if not specs:
        raise ValueError("Cannot repeat an empty object list")
    return [specs[index % len(specs)] for index in range(count)]


def _category_balanced(
    specs: list[ObjectSpec], count: int, *, offset: int = 0
) -> list[ObjectSpec]:
    by_category: dict[str, list[ObjectSpec]] = {}
    for spec in specs:
        by_category.setdefault(str(spec.category), []).append(spec)
    for category in by_category:
        by_category[category].sort(key=lambda spec: (_rank(spec.stable_id), spec.stable_id))
    categories = sorted(by_category, key=lambda item: (_rank(item), item))
    cursors = {category: 0 for category in categories}
    selected: list[ObjectSpec] = []
    skipped = 0
    while len(selected) < count:
        progressed = False
        for category in categories:
            cursor = cursors[category]
            values = by_category[category]
            if cursor >= len(values):
                continue
            spec = values[cursor]
            cursors[category] += 1
            progressed = True
            if skipped < offset:
                skipped += 1
                continue
            selected.append(spec)
            if len(selected) == count:
                break
        if not progressed:
            raise RuntimeError(
                f"Only {len(selected)} objects available after offset {offset}; requested {count}"
            )
    return selected


def _list_category_instances(
    source: InternDataAssetsSource,
    category: str,
    *,
    attempts: int,
    delay_seconds: float,
) -> list[str]:
    cache_existed = source._cache_path(category).is_file()
    for attempt in range(attempts):
        try:
            values = sorted(
                str(item) for item in source._list_category_instances(category)
            )
        except RuntimeError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_seconds * (attempt + 1))
            continue
        if not cache_existed:
            time.sleep(delay_seconds)
        return values
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--api-retries", type=int, default=8)
    parser.add_argument("--api-delay-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_workers <= 0:
        raise ValueError("max-workers must be positive")
    if args.api_retries <= 0:
        raise ValueError("api-retries must be positive")
    if args.api_delay_seconds < 0:
        raise ValueError("api-delay-seconds must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ycb_ids = sorted(str(item) for item in YCBSource().model_ids)
    if len(ycb_ids) != 74:
        raise RuntimeError(f"Expected 74 YCB IDs, found {len(ycb_ids)}")
    ycb_split = _split(ycb_ids, (50, 8, 16))

    source = InternDataAssetsSource()
    categories = sorted(source._list_categories())
    if len(categories) != 106:
        raise RuntimeError(f"Expected 106 InternData categories, found {len(categories)}")
    category_split = _split(categories, (80, 10, 16))
    category_instances = {
        category: _list_category_instances(
            source,
            category,
            attempts=args.api_retries,
            delay_seconds=args.api_delay_seconds,
        )
        for category in categories
    }
    all_instances = [
        (category, object_id)
        for category in categories
        for object_id in category_instances[category]
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        audit_rows = list(executor.map(_audit_interndata, all_instances))
    audit_rows.sort(key=lambda row: row.stable_id)

    train_instances: list[ObjectSpec] = []
    validation_instances: list[ObjectSpec] = []
    formal_unseen_instances: list[ObjectSpec] = []
    formal_unseen_categories: list[ObjectSpec] = []
    for category in category_split["train"]:
        ids = sorted(category_instances[category], key=lambda item: (_rank(f"{category}/{item}"), item))
        if len(ids) < 3:
            raise RuntimeError(f"Category {category!r} has fewer than three instances")
        train_ids, validation_id, formal_id = ids[:-2], ids[-2], ids[-1]
        train_instances.extend(
            ObjectSpec("interndata", object_id, category=category) for object_id in train_ids
        )
        validation_instances.append(ObjectSpec("interndata", validation_id, category=category))
        formal_unseen_instances.append(ObjectSpec("interndata", formal_id, category=category))
    for category in category_split["validation"]:
        validation_instances.extend(
            ObjectSpec("interndata", object_id, category=category)
            for object_id in category_instances[category]
        )
    for category in category_split["formal"]:
        formal_unseen_categories.extend(
            ObjectSpec("interndata", object_id, category=category)
            for object_id in category_instances[category]
        )

    manifest: dict[str, Any] = {
        "version": 1,
        "split_seed": SPLIT_SEED,
        "ycb": ycb_split,
        "interndata_categories": category_split,
        "interndata_instances": {
            "train": [spec.to_dict() for spec in train_instances],
            "validation": [spec.to_dict() for spec in validation_instances],
            "formal_unseen_instance": [spec.to_dict() for spec in formal_unseen_instances],
            "formal_unseen_category": [spec.to_dict() for spec in formal_unseen_categories],
        },
        "counts": {
            "ycb": len(ycb_ids),
            "interndata_categories": len(categories),
            "interndata_instances": len(all_instances),
            "interndata_loadable": sum(row.status == "loadable" for row in audit_rows),
            "interndata_not_cached": sum(row.status == "not_cached" for row in audit_rows),
            "interndata_invalid_mesh": sum(row.status == "invalid_mesh" for row in audit_rows),
        },
        "asset_root": str(ASSET_DIR),
    }
    (args.output_dir / "object_splits.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "static_asset_audit.jsonl").open("w") as output:
        for row in audit_rows:
            output.write(json.dumps(dataclasses.asdict(row), sort_keys=True) + "\n")

    ycb_train = [ObjectSpec("ycb", object_id) for object_id in ycb_split["train"]]
    ycb_validation = [ObjectSpec("ycb", object_id) for object_id in ycb_split["validation"]]
    ycb_formal = [ObjectSpec("ycb", object_id) for object_id in ycb_split["formal"]]
    train_dynamic = _category_balanced(train_instances, 1500)
    reserve_dynamic = _category_balanced(train_instances, 300, offset=1500)
    object_manifests = {
        "train_candidates": (
            _cube_specs(100, "train")
            + _repeat_specs(ycb_train, 200)
            + train_dynamic
        ),
        "train_reserve": reserve_dynamic,
        "validation_100": (
            _cube_specs(10, "validation")
            + _repeat_specs(ycb_validation, 30)
            + _category_balanced(validation_instances, 60)
        ),
        "formal_256": (
            _cube_specs(32, "formal")
            + _repeat_specs(ycb_formal, 64)
            + _category_balanced(formal_unseen_instances, 80)
            + _category_balanced(formal_unseen_categories, 80)
        ),
    }
    for name, specs in object_manifests.items():
        (args.output_dir / f"{name}_objects.json").write_text(
            json.dumps(
                {"objects": [spec.to_dict() for spec in specs]},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
