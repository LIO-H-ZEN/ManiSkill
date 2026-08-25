"""Pure benchmark aggregation for paired LiftAnything provider comparisons."""

from __future__ import annotations

import collections
from typing import Any, Mapping, Sequence

import numpy as np

K_VALUES = (1, 4, 8, 16)


def _success_at_k(row: Mapping[str, Any], k: int) -> bool:
    return bool(row["accepted"] and int(row["attempted_candidates"]) <= k)


def _stage_recall(row: Mapping[str, Any], k: int, field: str) -> bool:
    evaluations = row.get("candidate_evaluations", ())
    return any(
        int(item["rank"]) <= k and bool(item.get(field, False)) for item in evaluations
    )


def summarize_group(
    rows: Sequence[Mapping[str, Any]],
    episode_to_object: Mapping[str, str],
    object_categories: Mapping[str, str],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("benchmark group has no rows")
    per_object: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        episode_id = str(row["stable_episode_id"])
        if episode_id not in episode_to_object:
            raise ValueError(f"Unknown benchmark episode: {episode_id}")
        per_object[episode_to_object[episode_id]].append(row)
    object_success = {
        object_id: float(np.mean([bool(row["accepted"]) for row in object_rows]))
        for object_id, object_rows in per_object.items()
    }
    category_success: dict[str, list[float]] = collections.defaultdict(list)
    for object_id, success in object_success.items():
        category_success[object_categories[object_id]].append(success)
    return {
        "episodes": len(rows),
        "objects": len(per_object),
        "end_to_end_lift_success": float(
            np.mean([bool(row["accepted"]) for row in rows])
        ),
        "object_macro_robust_success": float(np.mean(list(object_success.values()))),
        "category_macro_robust_success": float(
            np.mean([np.mean(values) for values in category_success.values()])
        ),
        "success_at_k": {
            str(k): float(np.mean([_success_at_k(row, k) for row in rows]))
            for k in K_VALUES
        },
        "geometry_feasible_recall_at_k": {
            str(k): float(
                np.mean([_stage_recall(row, k, "geometry_feasible") for row in rows])
            )
            for k in K_VALUES
        },
        "ik_feasible_recall_at_k": {
            str(k): float(
                np.mean([_stage_recall(row, k, "ik_feasible") for row in rows])
            )
            for k in K_VALUES
        },
        "path_feasible_recall_at_k": {
            str(k): float(
                np.mean([_stage_recall(row, k, "path_feasible") for row in rows])
            )
            for k in K_VALUES
        },
        "grasp_oracle_status": "not_run",
        "grasp_oracle_recall_at_k": {str(k): None for k in K_VALUES},
        "object_success": object_success,
        "failure_distribution": dict(
            sorted(
                collections.Counter(
                    row["reason"] for row in rows if not row["accepted"]
                ).items()
            )
        ),
        "candidate_failure_stage_distribution": dict(
            sorted(
                collections.Counter(
                    str(item["failure_stage"])
                    for row in rows
                    for item in row.get("candidate_evaluations", ())
                    if item.get("failure_stage") is not None
                ).items()
            )
        ),
    }


def paired_bootstrap_interval(
    first: Mapping[str, float],
    second: Mapping[str, float],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    if set(first) != set(second) or not first:
        raise ValueError("paired bootstrap requires identical non-empty object IDs")
    object_ids = sorted(first)
    differences = np.asarray([second[key] - first[key] for key in object_ids])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    return (
        float(differences.mean()),
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    )


def analyze_benchmark(
    group_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    episode_to_object: Mapping[str, str],
    object_categories: Mapping[str, str],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    summaries = {
        group: summarize_group(rows, episode_to_object, object_categories)
        for group, rows in group_rows.items()
    }
    result: dict[str, Any] = {"groups": summaries}
    if "A" in summaries and "B" in summaries:
        mean, lower, upper = paired_bootstrap_interval(
            summaries["A"]["object_success"],
            summaries["B"]["object_success"],
            samples=bootstrap_samples,
            seed=seed,
        )
        simple_ids = [
            object_id
            for object_id, category in object_categories.items()
            if category == "simple_convex"
            and object_id in summaries["A"]["object_success"]
        ]
        simple_interval = None
        if simple_ids:
            simple_interval = paired_bootstrap_interval(
                {key: summaries["A"]["object_success"][key] for key in simple_ids},
                {key: summaries["B"]["object_success"][key] for key in simple_ids},
                samples=bootstrap_samples,
                seed=seed + 1,
            )
        result["paired_antipodal_minus_obb"] = {
            "mean": mean,
            "ci95": [lower, upper],
            "simple_convex_ci95": (
                None
                if simple_interval is None
                else [simple_interval[1], simple_interval[2]]
            ),
            "promotion_passed": bool(
                mean >= 0.10
                and lower > 0.0
                and simple_interval is not None
                and simple_interval[1] > -0.02
            ),
        }
    return result
