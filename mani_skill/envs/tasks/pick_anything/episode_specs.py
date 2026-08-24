"""Serializable contracts for deterministic LiftAnything episodes."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Literal, Mapping, Sequence

import numpy as np


ObjectSourceName = Literal["cube", "ycb", "interndata"]


def _finite_tuple(value: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != length or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite values, got {value!r}")
    return result


@dataclasses.dataclass(frozen=True)
class ObjectSpec:
    source: ObjectSourceName
    object_id: str
    category: str | None = None
    cube_half_size: float | None = None
    cube_color: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.source not in ("cube", "ycb", "interndata"):
            raise ValueError(f"Unsupported object source: {self.source!r}")
        if not self.object_id:
            raise ValueError("object_id must not be empty")
        if self.source == "cube":
            if self.category is not None:
                raise ValueError("cube ObjectSpec must not define category")
            if self.cube_half_size is None or not 0.005 <= self.cube_half_size <= 0.1:
                raise ValueError("cube_half_size must be in [0.005, 0.1] m")
            if self.cube_color is None:
                raise ValueError("cube ObjectSpec requires cube_color")
            color = _finite_tuple(self.cube_color, 4, "cube_color")
            if any(channel < 0.0 or channel > 1.0 for channel in color):
                raise ValueError("cube_color channels must be in [0, 1]")
            object.__setattr__(self, "cube_color", color)
        else:
            if self.cube_half_size is not None or self.cube_color is not None:
                raise ValueError(f"{self.source} ObjectSpec must not define cube fields")
            if self.source == "interndata" and not self.category:
                raise ValueError("interndata ObjectSpec requires category")
            if self.source == "ycb" and self.category is not None:
                raise ValueError("ycb ObjectSpec must not define category")

    @property
    def stable_id(self) -> str:
        if self.source == "interndata":
            return f"interndata/{self.category}/{self.object_id}"
        return f"{self.source}/{self.object_id}"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObjectSpec":
        allowed = {field.name for field in dataclasses.fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown ObjectSpec fields: {sorted(unknown)}")
        payload = dict(value)
        if payload.get("cube_color") is not None:
            payload["cube_color"] = tuple(payload["cube_color"])
        return cls(**payload)


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    stable_episode_id: str
    environment_seed: int
    object_spec: ObjectSpec
    object_position: tuple[float, float, float]
    object_quaternion: tuple[float, float, float, float]
    table_kind: str
    table_texture: str | None
    table_friction: tuple[float, float, float] | None
    floor_texture: str | None
    hdri: str | None
    directional_light_direction: tuple[float, float, float]
    directional_light_intensity: float
    robot_init_qpos: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.stable_episode_id:
            raise ValueError("stable_episode_id must not be empty")
        if isinstance(self.environment_seed, bool) or self.environment_seed < 0:
            raise ValueError("environment_seed must be a non-negative integer")
        object.__setattr__(
            self,
            "object_position",
            _finite_tuple(self.object_position, 3, "object_position"),
        )
        quaternion = _finite_tuple(self.object_quaternion, 4, "object_quaternion")
        if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-5):
            raise ValueError("object_quaternion must be normalized")
        object.__setattr__(self, "object_quaternion", quaternion)
        if self.table_friction is not None:
            object.__setattr__(
                self,
                "table_friction",
                _finite_tuple(self.table_friction, 3, "table_friction"),
            )
        direction = _finite_tuple(
            self.directional_light_direction, 3, "directional_light_direction"
        )
        if np.linalg.norm(direction) == 0.0:
            raise ValueError("directional_light_direction must be non-zero")
        object.__setattr__(self, "directional_light_direction", direction)
        if not np.isfinite(self.directional_light_intensity) or self.directional_light_intensity <= 0:
            raise ValueError("directional_light_intensity must be positive")
        qpos = _finite_tuple(self.robot_init_qpos, 8, "robot_init_qpos")
        object.__setattr__(self, "robot_init_qpos", qpos)

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["object_spec"] = self.object_spec.to_dict()
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeSpec":
        allowed = {field.name for field in dataclasses.fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown EpisodeSpec fields: {sorted(unknown)}")
        payload = dict(value)
        payload["object_spec"] = ObjectSpec.from_dict(payload["object_spec"])
        for key in (
            "object_position",
            "object_quaternion",
            "directional_light_direction",
            "robot_init_qpos",
        ):
            payload[key] = tuple(payload[key])
        if payload.get("table_friction") is not None:
            payload["table_friction"] = tuple(payload["table_friction"])
        return cls(**payload)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
