"""Strict RoboDojo General Pickup layout contracts for ManiSkill replay."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .episode_specs import ObjectSpec

CONTRACT_VERSION = "robodojo_general_pickup_piper_mvp_v1"
CONTRACT_ROOT_ENV = "MANISKILL_ROBODOJO_CONTRACT_ROOT"


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tuple(value: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != length or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite values, got {value!r}")
    return result


def _quaternion(value: Sequence[float], name: str) -> tuple[float, float, float, float]:
    result = _finite_tuple(value, 4, name)
    if not np.isclose(np.linalg.norm(result), 1.0, atol=1e-5):
        raise ValueError(f"{name} must be normalized")
    return result


@dataclasses.dataclass(frozen=True)
class RoboDojoSceneObjectSpec:
    asset_type: str
    category: str
    category_idx: int
    label: str | None
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    scale: tuple[float, float, float]
    mass: float | None
    friction: float | None
    descriptions: tuple[str, ...]
    instruction_candidates: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.asset_type not in {"Rigid", "Clutter"}:
            raise ValueError(f"Unsupported RoboDojo asset type: {self.asset_type!r}")
        if not self.category:
            raise ValueError("RoboDojo object category must not be empty")
        if isinstance(self.category_idx, bool) or self.category_idx < 0:
            raise ValueError(f"Invalid RoboDojo category_idx: {self.category_idx!r}")
        object.__setattr__(
            self, "position", _finite_tuple(self.position, 3, "position")
        )
        object.__setattr__(
            self, "quaternion", _quaternion(self.quaternion, "quaternion")
        )
        scale = _finite_tuple(self.scale, 3, "scale")
        if any(value <= 0.0 for value in scale):
            raise ValueError(f"RoboDojo object scale must be positive: {scale}")
        object.__setattr__(self, "scale", scale)
        if self.mass is not None and (not np.isfinite(self.mass) or self.mass <= 0.0):
            raise ValueError(f"RoboDojo object mass must be positive: {self.mass}")
        if self.friction is not None and (
            not np.isfinite(self.friction) or self.friction < 0.0
        ):
            raise ValueError(
                f"RoboDojo object friction must be non-negative: {self.friction}"
            )
        object.__setattr__(
            self,
            "descriptions",
            tuple(value.strip() for value in self.descriptions if value.strip()),
        )
        object.__setattr__(
            self,
            "instruction_candidates",
            tuple(
                value.strip() for value in self.instruction_candidates if value.strip()
            ),
        )

    @property
    def object_id(self) -> str:
        return f"{self.category_idx:05d}"

    @property
    def asset_key(self) -> str:
        return f"{self.asset_type}/{self.category}/{self.object_id}"

    @property
    def object_spec(self) -> ObjectSpec:
        return ObjectSpec(
            source="robodojo", category=self.category, object_id=self.object_id
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RoboDojoSceneObjectSpec":
        pose = value.get("pose_maniskill")
        if not isinstance(pose, Mapping):
            raise ValueError("RoboDojo object is missing pose_maniskill")
        physics = value.get("physics", {})
        if not isinstance(physics, Mapping):
            raise ValueError("RoboDojo object physics must be a mapping")
        return cls(
            asset_type=str(value["asset_type"]),
            category=str(value["category"]),
            category_idx=int(value["category_idx"]),
            label=value.get("label"),
            position=tuple(pose["position"]),
            quaternion=tuple(pose["orientation_wxyz"]),
            scale=tuple(value["scale"]),
            mass=None if physics.get("mass") is None else float(physics["mass"]),
            friction=(
                None if physics.get("friction") is None else float(physics["friction"])
            ),
            descriptions=tuple(str(item) for item in value.get("descriptions", [])),
            instruction_candidates=tuple(
                str(item) for item in value.get("instruction_candidates", [])
            ),
        )


@dataclasses.dataclass(frozen=True)
class RoboDojoLayoutSpec:
    filtered_layout_id: int
    contract_sha256: str
    source_layout_path: str
    source_layout_sha256: str
    expected_clutter_count: int
    source_clutter_count: int
    target: RoboDojoSceneObjectSpec
    clutter: tuple[RoboDojoSceneObjectSpec, ...]

    def __post_init__(self) -> None:
        if self.target.label != "target":
            raise ValueError("RoboDojo General Pickup contract requires one target")
        if not self.target.instruction_candidates:
            raise ValueError("RoboDojo target requires at least one task instruction")
        if self.source_clutter_count != len(self.clutter):
            raise ValueError(
                "source_clutter_count does not match the serialized clutter list"
            )
        if self.source_clutter_count < 1:
            raise ValueError("RoboDojo General Pickup layout requires clutter")
        categories = [item.category for item in self.clutter]
        if len(categories) != len(set(categories)):
            raise ValueError("RoboDojo clutter categories must be unique")

    @property
    def prompt(self) -> str:
        return self.target.instruction_candidates[0]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RoboDojoLayoutSpec":
        if value.get("contract_version") != CONTRACT_VERSION:
            raise ValueError(
                f"Unsupported RoboDojo contract version: {value.get('contract_version')!r}"
            )
        payload = dict(value)
        stored_hash = payload.pop("contract_sha256", None)
        actual_hash = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        if stored_hash != actual_hash:
            raise RuntimeError("RoboDojo layout contract SHA-256 mismatch")
        source_layout = value.get("source_layout")
        if not isinstance(source_layout, Mapping):
            raise ValueError("RoboDojo contract is missing source_layout")
        target = RoboDojoSceneObjectSpec.from_dict(value["target"])
        clutter = tuple(
            RoboDojoSceneObjectSpec.from_dict(item) for item in value["clutter"]
        )
        return cls(
            filtered_layout_id=int(value["filtered_layout_id"]),
            contract_sha256=str(stored_hash),
            source_layout_path=str(source_layout["path"]),
            source_layout_sha256=str(source_layout["sha256"]),
            expected_clutter_count=int(value["expected_clutter_count"]),
            source_clutter_count=int(value["source_clutter_count"]),
            target=target,
            clutter=clutter,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RoboDojoLayoutSpec":
        contract_path = Path(path).expanduser().resolve()
        if not contract_path.is_file():
            raise FileNotFoundError(contract_path)
        value = json.loads(contract_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"Expected a JSON object: {contract_path}")
        return cls.from_dict(value)


def load_robodojo_layout_spec(
    *,
    layout_id: int,
    contract_root: str | Path | None = None,
    asset_root: str | Path | None = None,
) -> RoboDojoLayoutSpec:
    """Load one contract and bind it to the exported and converted manifests."""

    root_value = (
        contract_root
        if contract_root is not None
        else os.environ.get(CONTRACT_ROOT_ENV)
    )
    if root_value is None:
        raise ValueError(
            "RoboDojo contracts require an explicit root or " f"{CONTRACT_ROOT_ENV}"
        )
    root = Path(root_value).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_manifest_hash = manifest.pop("manifest_sha256", None)
    actual_manifest_hash = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    if stored_manifest_hash != actual_manifest_hash:
        raise RuntimeError(f"RoboDojo contract manifest hash mismatch: {manifest_path}")
    rows = {
        int(row["filtered_layout_id"]): row for row in manifest.get("contracts", [])
    }
    if layout_id not in rows:
        raise KeyError(f"RoboDojo filtered layout ID is not exported: {layout_id}")
    row = rows[layout_id]
    contract_path = root / row["contract_file"]
    spec = RoboDojoLayoutSpec.load(contract_path)
    if spec.contract_sha256 != row["contract_sha256"]:
        raise RuntimeError(
            f"RoboDojo manifest/contract hash mismatch for layout {layout_id}"
        )
    if asset_root is not None:
        conversion_manifest_path = (
            Path(asset_root).expanduser().resolve() / "manifest.json"
        )
        if not conversion_manifest_path.is_file():
            raise FileNotFoundError(conversion_manifest_path)
        conversion_manifest = json.loads(
            conversion_manifest_path.read_text(encoding="utf-8")
        )
        expected_conversion_hash = manifest.get("conversion_manifest_sha256")
        if expected_conversion_hash is not None:
            if expected_conversion_hash != _sha256_file(conversion_manifest_path):
                raise RuntimeError(
                    "RoboDojo training contracts reference a different conversion manifest"
                )
        elif conversion_manifest.get("contract_manifest_sha256") != _sha256_file(
            manifest_path
        ):
            raise RuntimeError(
                "RoboDojo converted assets were built from a different contract manifest"
            )
    return spec
