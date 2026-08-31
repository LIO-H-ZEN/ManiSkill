"""Deterministic tabletop grasp proposals from collision-mesh cross sections."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import trimesh

from .contracts import GraspCandidate, GraspProviderName
from .geometry import ResolvedObjectGeometry

TABLETOP_PROVIDER_VERSION = "tabletop_cross_section_v4_configurable_offsets"
TABLETOP_OFFSET_LATERAL_FRACTIONS = (-0.3, 0.3, -0.45, 0.45)


@dataclasses.dataclass(frozen=True)
class TabletopConfig:
    yaw_samples: int = 24
    height_fractions: tuple[float, ...] = (0.35, 0.45, 0.5, 0.65)
    lateral_fractions: tuple[float, ...] = (0.0, -0.15, 0.15)
    approach_tilts_degrees: tuple[float, ...] = (7.5, 0.0, 15.0)
    approach_azimuth_offsets_degrees: tuple[float, ...] = (20.0, 0.0, -20.0)
    minimum_width: float = 0.001
    maximum_width: float = 0.068
    normal_error_degrees: float = 35.0
    maximum_candidates: int = 256
    intersection_merge_tolerance: float = 1e-5
    pose_translation_nms: float = 0.004
    pose_rotation_nms_degrees: float = 3.0
    contact_region_size: float = 0.015

    def __post_init__(self) -> None:
        if self.yaw_samples < 2:
            raise ValueError("yaw_samples must be at least two")
        if not self.height_fractions or not self.lateral_fractions:
            raise ValueError("height and lateral fractions must not be empty")
        if any(not 0.0 <= value <= 1.0 for value in self.height_fractions):
            raise ValueError("height fractions must lie in [0, 1]")
        if any(abs(value) > 0.5 for value in self.lateral_fractions):
            raise ValueError("lateral fractions must lie in [-0.5, 0.5]")
        if any(not 0.0 <= value <= 45.0 for value in self.approach_tilts_degrees):
            raise ValueError("approach tilts must lie in [0, 45] degrees")
        if any(
            not -90.0 <= value <= 90.0
            for value in self.approach_azimuth_offsets_degrees
        ):
            raise ValueError("approach azimuth offsets must lie in [-90, 90] degrees")
        if not 0.001 <= self.minimum_width < self.maximum_width <= 0.068:
            raise ValueError("width range must lie within [0.001, 0.068] m")
        if not 0.0 < self.normal_error_degrees < 90.0:
            raise ValueError("normal_error_degrees must lie in (0, 90)")
        if self.maximum_candidates <= 0:
            raise ValueError("maximum_candidates must be positive")


@dataclasses.dataclass(frozen=True)
class _LineHit:
    distance: float
    point: np.ndarray
    normal: np.ndarray


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if result.shape != (3,) or not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name} must be a finite non-zero 3D vector")
    return result / norm


def _line_hits(
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    direction: np.ndarray,
    *,
    merge_tolerance: float,
) -> list[_LineHit]:
    """Intersect an infinite line with a mesh without requiring ``rtree``."""

    direction = _normalize(direction, name="line direction")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    projections = vertices @ direction
    span = float(np.ptp(projections)) + 0.02
    origin = np.asarray(center, dtype=np.float64) - direction * (
        span + float(np.max(projections) - center @ direction)
    )

    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    vertex0 = triangles[:, 0]
    edge1 = triangles[:, 1] - vertex0
    edge2 = triangles[:, 2] - vertex0
    repeated_direction = np.broadcast_to(direction, edge2.shape)
    h = np.cross(repeated_direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(determinant) > 1e-12
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    displacement = origin - vertex0
    u = inverse * np.einsum("ij,ij->i", displacement, h)
    q = np.cross(displacement, edge1)
    v = inverse * np.einsum("ij,ij->i", repeated_direction, q)
    distance = inverse * np.einsum("ij,ij->i", edge2, q)
    valid &= u >= -1e-9
    valid &= v >= -1e-9
    valid &= u + v <= 1.0 + 1e-9
    valid &= distance >= 0.0
    face_indices = np.flatnonzero(valid)
    if len(face_indices) == 0:
        return []

    rows = sorted(
        [
            (
                float((origin + direction * float(distance[index])) @ direction),
                origin + direction * float(distance[index]),
                np.asarray(mesh.face_normals[index], dtype=np.float64),
            )
            for index in face_indices
        ],
        key=lambda row: row[0],
    )
    hits: list[_LineHit] = []
    group: list[tuple[float, np.ndarray, np.ndarray]] = []
    for row in rows:
        if group and row[0] - group[-1][0] > merge_tolerance:
            hits.append(_merge_hit_group(group))
            group = []
        group.append(row)
    if group:
        hits.append(_merge_hit_group(group))
    return hits


def _union_line_hits(
    meshes: tuple[trimesh.Trimesh, ...],
    center: np.ndarray,
    direction: np.ndarray,
    *,
    merge_tolerance: float,
) -> list[_LineHit]:
    """Return the exterior line intersections of a union of convex components."""

    direction = _normalize(direction, name="line direction")
    events: list[tuple[_LineHit, bool, tuple[int, int]]] = []
    for mesh_index, mesh in enumerate(meshes):
        hits = _line_hits(
            mesh,
            center,
            direction,
            merge_tolerance=merge_tolerance,
        )
        if len(hits) % 2:
            continue
        for pair_index, (entry, exit_) in enumerate(zip(hits[0::2], hits[1::2])):
            interval_id = (mesh_index, pair_index)
            events.append((entry, True, interval_id))
            events.append((exit_, False, interval_id))
    if not events:
        return []
    events.sort(key=lambda event: event[0].distance)

    union_hits = []
    active: set[tuple[int, int]] = set()
    group: list[tuple[_LineHit, bool, tuple[int, int]]] = []

    def flush() -> None:
        if not group:
            return
        was_inside = bool(active)
        for _, entering, interval_id in group:
            if not entering:
                active.discard(interval_id)
        for _, entering, interval_id in group:
            if entering:
                active.add(interval_id)
        is_inside = bool(active)
        if was_inside == is_inside:
            return
        boundary = [
            hit
            for hit, entering, _ in group
            if entering == (not was_inside and is_inside)
        ]
        rows = [(hit.distance, hit.point, hit.normal) for hit in boundary]
        union_hits.append(_merge_hit_group(rows))

    for event in events:
        if group and event[0].distance - group[-1][0].distance > merge_tolerance:
            flush()
            group = []
        group.append(event)
    flush()
    return union_hits


def _merge_hit_group(rows: list[tuple[float, np.ndarray, np.ndarray]]) -> _LineHit:
    normal = np.mean([row[2] for row in rows], axis=0)
    normal = _normalize(normal, name="merged surface normal")
    return _LineHit(
        distance=float(np.mean([row[0] for row in rows])),
        point=np.mean([row[1] for row in rows], axis=0),
        normal=normal,
    )


def _rotation_distance_with_symmetry(first: np.ndarray, second: np.ndarray) -> float:
    symmetry = np.diag([-1.0, -1.0, 1.0])

    def angle(delta: np.ndarray) -> float:
        cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
        return math.acos(cosine)

    return min(angle(first.T @ second), angle(first.T @ second @ symmetry))


def _is_duplicate(
    transform: np.ndarray,
    selected: list[GraspCandidate],
    config: TabletopConfig,
) -> bool:
    angle_limit = math.radians(config.pose_rotation_nms_degrees)
    return any(
        np.linalg.norm(transform[:3, 3] - item.object_T_tcp[:3, 3])
        <= config.pose_translation_nms
        and _rotation_distance_with_symmetry(
            transform[:3, :3], item.object_T_tcp[:3, :3]
        )
        <= angle_limit
        for item in selected
    )


def _approach_directions(
    object_world_center: np.ndarray,
    robot_base_position: np.ndarray,
    tilts_degrees: tuple[float, ...],
    azimuth_offsets_degrees: tuple[float, ...],
) -> list[tuple[float, float, np.ndarray]]:
    planar = object_world_center[:2] - robot_base_position[:2]
    planar_norm = float(np.linalg.norm(planar))
    if planar_norm <= 1e-9:
        raise ValueError("object center must not coincide with the robot base axis")
    radial = planar / planar_norm
    approaches = []
    for tilt_degrees in tilts_degrees:
        tilt = math.radians(float(tilt_degrees))
        offsets = (0.0,) if tilt_degrees == 0.0 else azimuth_offsets_degrees
        for azimuth_degrees in offsets:
            azimuth = math.radians(float(azimuth_degrees))
            direction = np.array(
                [
                    (radial[0] * math.cos(azimuth) - radial[1] * math.sin(azimuth))
                    * math.sin(tilt),
                    (radial[0] * math.sin(azimuth) + radial[1] * math.cos(azimuth))
                    * math.sin(tilt),
                    -math.cos(tilt),
                ],
                dtype=np.float64,
            )
            approaches.append((float(tilt_degrees), float(azimuth_degrees), direction))
    return approaches


class TabletopGraspProvider:
    """Generate table-aware side pinches through deterministic mesh slices."""

    def __init__(self, config: TabletopConfig | None = None):
        self.config = config or TabletopConfig()

    def generate(
        self,
        geometry: ResolvedObjectGeometry,
        *,
        world_T_object: np.ndarray,
        robot_base_position: np.ndarray,
    ) -> list[GraspCandidate]:
        if not geometry.collision_meshes:
            raise ValueError("tabletop provider requires collision geometry")
        world_T_object = np.asarray(world_T_object, dtype=np.float64)
        if world_T_object.shape != (4, 4) or not np.all(np.isfinite(world_T_object)):
            raise ValueError("world_T_object must be a finite 4x4 transform")
        rotation = world_T_object[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("world_T_object rotation must be orthonormal")
        robot_base_position = np.asarray(robot_base_position, dtype=np.float64)
        if robot_base_position.shape != (3,) or not np.all(
            np.isfinite(robot_base_position)
        ):
            raise ValueError("robot_base_position must be a finite 3D point")

        mesh = trimesh.util.concatenate(
            [item.copy() for item in geometry.collision_meshes]
        )
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError("collision geometry produced an empty mesh")
        center = np.asarray(mesh.center_mass, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            center = np.asarray(mesh.centroid, dtype=np.float64)
        world_center = rotation @ center + world_T_object[:3, 3]
        world_up_local = _normalize(rotation.T @ np.array([0.0, 0.0, 1.0]), name="up")
        approaches_world = _approach_directions(
            world_center,
            robot_base_position,
            self.config.approach_tilts_degrees,
            self.config.approach_azimuth_offsets_degrees,
        )
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        up_projection = vertices @ world_up_local
        up_low = float(up_projection.min())
        up_high = float(up_projection.max())
        center_up = float(center @ world_up_local)
        cosine_limit = math.cos(math.radians(self.config.normal_error_degrees))

        proposals = []
        pair_index = 0
        for yaw_index in range(self.config.yaw_samples):
            yaw = math.pi * yaw_index / self.config.yaw_samples
            closing_world = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            closing_local = _normalize(rotation.T @ closing_world, name="closing")
            lateral_local = _normalize(
                np.cross(world_up_local, closing_local), name="lateral"
            )
            lateral_projection = vertices @ lateral_local
            lateral_extent = float(np.ptp(lateral_projection))
            for height_fraction in self.config.height_fractions:
                target_up = up_low + float(height_fraction) * (up_high - up_low)
                for lateral_fraction in self.config.lateral_fractions:
                    line_center = center.copy()
                    line_center += world_up_local * (target_up - center_up)
                    line_center += lateral_local * (
                        float(lateral_fraction) * lateral_extent
                    )
                    hits = _union_line_hits(
                        geometry.collision_meshes,
                        line_center,
                        closing_local,
                        merge_tolerance=self.config.intersection_merge_tolerance,
                    )
                    for first, second in zip(hits[0::2], hits[1::2]):
                        width = float(second.distance - first.distance)
                        if (
                            not self.config.minimum_width
                            <= width
                            <= self.config.maximum_width
                        ):
                            continue
                        first_alignment = float(-first.normal @ closing_local)
                        second_alignment = float(second.normal @ closing_local)
                        if (
                            first_alignment < cosine_limit
                            or second_alignment < cosine_limit
                        ):
                            continue
                        midpoint = (first.point + second.point) * 0.5
                        midpoint_to_com = float(np.linalg.norm(midpoint - center))
                        normal_score = 0.5 * (first_alignment + second_alignment)
                        for approach_index, (
                            tilt_degrees,
                            azimuth_degrees,
                            approach_world,
                        ) in enumerate(approaches_world):
                            approach_local = _normalize(
                                rotation.T @ approach_world, name="approach"
                            )
                            closing = closing_local - approach_local * float(
                                closing_local @ approach_local
                            )
                            closing = _normalize(closing, name="projected closing")
                            ortho = _normalize(
                                np.cross(closing, approach_local), name="orthogonal"
                            )
                            transform = np.eye(4)
                            transform[:3, :3] = np.column_stack(
                                [ortho, closing, approach_local]
                            )
                            transform[:3, 3] = midpoint
                            priority = (
                                -round(normal_score, 6),
                                midpoint_to_com,
                                abs(float(lateral_fraction)),
                                abs(float(height_fraction) - 0.5),
                                approach_index,
                                yaw_index,
                                pair_index,
                            )
                            proposals.append(
                                (
                                    priority,
                                    pair_index,
                                    yaw_index,
                                    height_fraction,
                                    lateral_fraction,
                                    approach_index,
                                    tilt_degrees,
                                    azimuth_degrees,
                                    transform,
                                    width,
                                    normal_score,
                                    midpoint_to_com,
                                    first.point,
                                    second.point,
                                )
                            )
                        pair_index += 1

        proposals.sort(key=lambda row: row[0])
        selected: list[GraspCandidate] = []
        for (
            _,
            contact_pair_index,
            yaw_index,
            height_fraction,
            lateral_fraction,
            approach_index,
            tilt_degrees,
            azimuth_degrees,
            transform,
            width,
            score,
            midpoint_to_com,
            first_point,
            second_point,
        ) in proposals:
            if _is_duplicate(transform, selected, self.config):
                continue
            midpoint = transform[:3, 3]
            candidate = GraspCandidate(
                candidate_id=(
                    f"slice-{contact_pair_index:04d}-yaw-{yaw_index:02d}-"
                    f"h-{height_fraction:.2f}-lat-{lateral_fraction:+.2f}-"
                    f"approach-{approach_index:02d}"
                ),
                source=GraspProviderName.TABLETOP,
                object_T_tcp=transform,
                required_width=width,
                proposal_score=score,
                contact_points=np.stack([first_point, second_point]),
                metadata={
                    "provider_version": TABLETOP_PROVIDER_VERSION,
                    "contact_region": tuple(
                        int(value)
                        for value in np.floor(
                            midpoint / self.config.contact_region_size
                        )
                    ),
                    "pair_index": contact_pair_index,
                    "yaw_index": yaw_index,
                    "height_fraction": float(height_fraction),
                    "lateral_fraction": float(lateral_fraction),
                    "tilt_degrees": tilt_degrees,
                    "azimuth_offset_degrees": azimuth_degrees,
                    "midpoint_to_com": midpoint_to_com,
                    "contact_surface": geometry.collision_source,
                },
            )
            selected.append(candidate)
            if len(selected) == self.config.maximum_candidates:
                break
        if not selected:
            raise RuntimeError(
                "proposal-empty: no tabletop mesh cross-section produced a valid pinch"
            )
        return selected
