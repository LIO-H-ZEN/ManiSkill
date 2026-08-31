"""Deterministic unseen-layout contracts for General Pickup training."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .general_pickup_specs import CONTRACT_VERSION, RoboDojoLayoutSpec

TRAINING_PROFILE = "mirror_train"
TRAINING_CONTRACT_GENERATOR_VERSION = "robodojo_mirror_train_v2"
PIPER_BASE_XY = (-0.35, 0.0)
PIPER_BASE_EXCLUSION_RADIUS = 0.08
DEFAULT_TARGET_XLIM = (-0.25, 0.25)
DEFAULT_TARGET_YLIM = (-0.25, 0.0)


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hashed_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    stored_hash = value.pop("manifest_sha256", None)
    actual_hash = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    if stored_hash != actual_hash:
        raise RuntimeError(f"manifest SHA-256 mismatch: {path}")
    value["manifest_sha256"] = stored_hash
    return value


def _asset_key(value: Mapping[str, Any]) -> str:
    return f"{value['asset_type']}/{value['category']}/{int(value['category_idx']):05d}"


def _object_radius(value: Mapping[str, Any]) -> float:
    geometry = value.get("metadata", {}).get("geometry", {})
    projection = (
        value.get("metadata", {})
        .get("active", {})
        .get("place", {})
        .get("up", {})
        .get("projection_circle", {})
    )
    radius = projection.get("radius")
    if radius is None:
        extents = geometry.get("oriented_bbox", {}).get("extents")
        if not isinstance(extents, Sequence) or len(extents) != 3:
            raise ValueError(f"object {_asset_key(value)} has no placement radius")
        radius = 0.5 * math.hypot(float(extents[0]), float(extents[1]))
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"object {_asset_key(value)} has invalid radius {radius}")
    return radius


def _quat_multiply(
    lhs: Sequence[float], rhs: Sequence[float]
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = (float(value) for value in lhs)
    rw, rx, ry, rz = (float(value) for value in rhs)
    result = np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("generated quaternion is invalid")
    return tuple(float(value) for value in result / norm)


def _yaw_quaternion(angle: float) -> tuple[float, float, float, float]:
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def _maniskill_to_robodojo_pose(
    position: Sequence[float], quaternion: Sequence[float]
) -> tuple[list[float], list[float]]:
    x, y, z = (float(value) for value in position)
    robodojo_position = [-y, x - 0.1, z + 0.765]
    robodojo_quaternion = _quat_multiply(_yaw_quaternion(math.pi / 2.0), quaternion)
    return robodojo_position, list(robodojo_quaternion)


def _placed_object(
    source: Mapping[str, Any], *, x: float, y: float, yaw_delta: float
) -> dict[str, Any]:
    result = copy.deepcopy(dict(source))
    original_pose = source["pose_maniskill"]
    z = float(original_pose["position"][2])
    quaternion = _quat_multiply(
        _yaw_quaternion(yaw_delta), original_pose["orientation_wxyz"]
    )
    position = [float(x), float(y), z]
    robodojo_position, robodojo_quaternion = _maniskill_to_robodojo_pose(
        position, quaternion
    )
    result["pose_maniskill"] = {
        "position": position,
        "orientation_wxyz": list(quaternion),
    }
    result["pose_robodojo"] = {
        "position": robodojo_position,
        "orientation_wxyz": robodojo_quaternion,
    }
    return result


def _sample_position(
    *,
    rng: np.random.Generator,
    radius: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    occupied: list[tuple[float, float, float]],
    margin: float,
    attempts: int = 10_000,
) -> tuple[float, float]:
    if xlim[0] >= xlim[1] or ylim[0] >= ylim[1]:
        raise ValueError(f"invalid placement bounds: {xlim=} {ylim=}")
    for _ in range(attempts):
        # RoboDojo's layout limits constrain actor origins, not the complete
        # projection circle. Large but valid objects such as the whisk would
        # otherwise be rejected before placement.
        x = float(rng.uniform(*xlim))
        y = float(rng.uniform(*ylim))
        if all(
            math.hypot(x - other_x, y - other_y) >= radius + other_radius + margin
            for other_x, other_y, other_radius in occupied
        ):
            return x, y
    raise RuntimeError(
        f"could not place radius={radius:.4f} in {xlim=} {ylim=} after {attempts} attempts"
    )


def _validated_bounds(
    name: str, bounds: Sequence[float]
) -> tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower, upper = (float(value) for value in bounds)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(f"invalid {name}: {(lower, upper)}")
    return lower, upper


def load_source_objects(
    contract_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = contract_root / "manifest.json"
    manifest = _load_hashed_json(manifest_path)
    targets: dict[str, dict[str, Any]] = {}
    clutter: dict[str, dict[str, Any]] = {}
    template: dict[str, Any] | None = None
    for row in manifest.get("contracts", []):
        contract_path = contract_root / row["contract_file"]
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        RoboDojoLayoutSpec.from_dict(payload)
        if payload["contract_sha256"] != row["contract_sha256"]:
            raise RuntimeError(f"manifest/contract hash mismatch: {contract_path}")
        if template is None:
            template = payload
        targets.setdefault(_asset_key(payload["target"]), payload["target"])
        for item in payload["clutter"]:
            clutter.setdefault(_asset_key(item), item)
    if not targets or not clutter or template is None:
        raise ValueError("source contracts must contain targets and clutter")
    return (
        [targets[key] for key in sorted(targets)],
        [clutter[key] for key in sorted(clutter)],
        template,
    )


def _select_clutter(
    *,
    rng: np.random.Generator,
    pool: Sequence[dict[str, Any]],
    target_category: str,
    count: int,
) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for item in pool:
        if item["category"] == target_category:
            continue
        by_category.setdefault(str(item["category"]), []).append(item)
    categories = sorted(by_category)
    if len(categories) < count:
        raise ValueError(
            f"need {count} unique clutter categories, found {len(categories)}"
        )
    selected_categories = rng.choice(categories, size=count, replace=False).tolist()
    return [
        by_category[category][int(rng.integers(len(by_category[category])))]
        for category in selected_categories
    ]


def generate_layout(
    *,
    layout_id: int,
    generation_seed: int,
    target_source: Mapping[str, Any],
    clutter_sources: Sequence[Mapping[str, Any]],
    template: Mapping[str, Any],
    margin: float,
    target_xlim: Sequence[float] = DEFAULT_TARGET_XLIM,
    target_ylim: Sequence[float] = DEFAULT_TARGET_YLIM,
) -> dict[str, Any]:
    if len(clutter_sources) != 10:
        raise ValueError("mirror_train requires exactly 10 clutter objects")
    target_xlim = _validated_bounds("target_xlim", target_xlim)
    target_ylim = _validated_bounds("target_ylim", target_ylim)
    rng = np.random.default_rng(generation_seed)
    occupied: list[tuple[float, float, float]] = [
        (*PIPER_BASE_XY, PIPER_BASE_EXCLUSION_RADIUS)
    ]
    target_radius = _object_radius(target_source)
    target_x, target_y = _sample_position(
        rng=rng,
        radius=target_radius,
        xlim=target_xlim,
        ylim=target_ylim,
        occupied=occupied,
        margin=margin,
    )
    target = _placed_object(
        target_source,
        x=target_x,
        y=target_y,
        yaw_delta=float(rng.uniform(-math.radians(20), math.radians(20))),
    )
    occupied.append((target_x, target_y, target_radius))

    placed_clutter: dict[int, dict[str, Any]] = {}
    indexed_sources = list(enumerate(clutter_sources))
    indexed_sources.sort(key=lambda pair: _object_radius(pair[1]), reverse=True)
    for original_index, source in indexed_sources:
        radius = _object_radius(source)
        x, y = _sample_position(
            rng=rng,
            radius=radius,
            xlim=(-0.45, 0.45),
            ylim=(-0.45, 0.35),
            occupied=occupied,
            margin=margin,
        )
        occupied.append((x, y, radius))
        placed_clutter[original_index] = _placed_object(
            source,
            x=x,
            y=y,
            yaw_delta=float(rng.uniform(-math.radians(30), math.radians(30))),
        )
    clutter = [placed_clutter[index] for index in range(len(clutter_sources))]

    generation_payload = {
        "generator_version": TRAINING_CONTRACT_GENERATOR_VERSION,
        "generation_seed": generation_seed,
        "layout_id": layout_id,
        "target_asset_key": _asset_key(target),
        "clutter_asset_keys": [_asset_key(item) for item in clutter],
        "target_pose": target["pose_maniskill"],
        "clutter_poses": [item["pose_maniskill"] for item in clutter],
    }
    payload = {
        "contract_version": CONTRACT_VERSION,
        "profile": TRAINING_PROFILE,
        "generator_version": TRAINING_CONTRACT_GENERATOR_VERSION,
        "generation_seed": generation_seed,
        "filtered_layout_id": layout_id,
        "source_layout": {
            "path": f"generated://{TRAINING_PROFILE}/{layout_id:06d}",
            "sha256": hashlib.sha256(
                canonical_json_bytes(generation_payload)
            ).hexdigest(),
        },
        "expected_clutter_count": 10,
        "source_clutter_count": 10,
        "source_clutter_count_matches_config": True,
        "target": target,
        "clutter": clutter,
        "fixtures": copy.deepcopy(template.get("fixtures", {})),
        "static_contract": copy.deepcopy(template.get("static_contract", {})),
    }
    payload["contract_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    RoboDojoLayoutSpec.from_dict(payload)
    return payload


def generate_training_contracts(
    *,
    source_contract_root: Path,
    converted_asset_root: Path,
    output_root: Path,
    count: int,
    seed: int,
    first_layout_id: int = 1000,
    margin: float = 0.02,
    target_xlim: Sequence[float] = DEFAULT_TARGET_XLIM,
    target_ylim: Sequence[float] = DEFAULT_TARGET_YLIM,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    if first_layout_id < 0:
        raise ValueError("first_layout_id must be non-negative")
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be finite and non-negative")
    target_xlim = _validated_bounds("target_xlim", target_xlim)
    target_ylim = _validated_bounds("target_ylim", target_ylim)
    if output_root.exists():
        raise FileExistsError(output_root)
    conversion_manifest_path = converted_asset_root / "manifest.json"
    conversion_manifest = _load_hashed_json(conversion_manifest_path)
    declared_assets = {row["asset_key"] for row in conversion_manifest["assets"]}
    targets, clutter_pool, template = load_source_objects(source_contract_root)
    required_assets = {
        *(_asset_key(item) for item in targets),
        *(_asset_key(item) for item in clutter_pool),
    }
    missing_assets = sorted(required_assets - declared_assets)
    if missing_assets:
        raise KeyError(f"converted asset manifest is missing {missing_assets}")

    output_root.mkdir(parents=True)
    master_rng = np.random.default_rng(seed)
    target_order = master_rng.permutation(len(targets)).tolist()
    rows = []
    for offset in range(count):
        layout_id = first_layout_id + offset
        target_source = targets[target_order[offset % len(target_order)]]
        generation_seed = int(master_rng.integers(0, 2**63 - 1))
        layout_rng = np.random.default_rng(generation_seed)
        last_error: Exception | None = None
        for _ in range(100):
            clutter_sources = _select_clutter(
                rng=layout_rng,
                pool=clutter_pool,
                target_category=str(target_source["category"]),
                count=10,
            )
            try:
                payload = generate_layout(
                    layout_id=layout_id,
                    generation_seed=generation_seed,
                    target_source=target_source,
                    clutter_sources=clutter_sources,
                    template=template,
                    margin=margin,
                    target_xlim=target_xlim,
                    target_ylim=target_ylim,
                )
            except RuntimeError as error:
                last_error = error
                generation_seed = int(layout_rng.integers(0, 2**63 - 1))
                continue
            break
        else:
            raise RuntimeError(f"failed to generate layout {layout_id}") from last_error
        filename = f"general_pickup_filtered_{layout_id:03d}.json"
        (output_root / filename).write_bytes(canonical_json_bytes(payload))
        rows.append(
            {
                "filtered_layout_id": layout_id,
                "contract_file": filename,
                "contract_sha256": payload["contract_sha256"],
                "generation_seed": payload["generation_seed"],
                "target": {
                    "category": payload["target"]["category"],
                    "category_idx": payload["target"]["category_idx"],
                },
                "clutter_asset_keys": [_asset_key(item) for item in payload["clutter"]],
            }
        )

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "profile": TRAINING_PROFILE,
        "generator_version": TRAINING_CONTRACT_GENERATOR_VERSION,
        "seed": seed,
        "layout_ids": [row["filtered_layout_id"] for row in rows],
        "source_contract_manifest_sha256": sha256_file(
            source_contract_root / "manifest.json"
        ),
        "conversion_manifest_sha256": sha256_file(conversion_manifest_path),
        "count": count,
        "first_layout_id": first_layout_id,
        "placement_margin": margin,
        "target_position_bounds": {
            "x": list(target_xlim),
            "y": list(target_ylim),
        },
        "contracts": rows,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    (output_root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest
