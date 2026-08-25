"""Deterministic mesh-based antipodal grasp proposal generation."""

from __future__ import annotations

import dataclasses
import hashlib
import math

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .contracts import GraspCandidate, GraspProviderName
from .geometry import ResolvedObjectGeometry

ANTIPODAL_PROVIDER_VERSION = "mesh_antipodal_v2"


@dataclasses.dataclass(frozen=True)
class AntipodalConfig:
    area_samples: int = 3072
    balanced_samples: int = 1024
    minimum_width: float = 0.001
    maximum_width: float = 0.068
    normal_error_degrees: float = 25.0
    maximum_opposites_per_point: int = 8
    raycast_sample_limit: int = 1024
    raycast_batch_size: int = 16
    maximum_contact_pairs: int = 2048
    approaches_per_pair: int = 12
    maximum_candidates: int = 256
    pair_midpoint_nms: float = 0.002
    pair_axis_nms_degrees: float = 10.0
    pose_translation_nms: float = 0.01
    pose_rotation_nms_degrees: float = 15.0
    contact_region_size: float = 0.015
    maximum_per_contact_region: int = 8

    def __post_init__(self) -> None:
        if self.area_samples <= 0 or self.balanced_samples <= 0:
            raise ValueError("surface sample counts must be positive")
        if not 0.001 <= self.minimum_width < self.maximum_width <= 0.068:
            raise ValueError("width range must lie within [0.001, 0.068] m")
        if not 0.0 < self.normal_error_degrees < 90.0:
            raise ValueError("normal_error_degrees must be in (0, 90)")
        if self.approaches_per_pair <= 0 or self.maximum_candidates <= 0:
            raise ValueError("candidate budgets must be positive")
        if self.raycast_sample_limit <= 0 or self.raycast_batch_size <= 0:
            raise ValueError("raycast budgets must be positive")

    @property
    def fingerprint(self) -> str:
        payload = repr(dataclasses.asdict(self)).encode()
        return hashlib.sha256(ANTIPODAL_PROVIDER_VERSION.encode() + payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class SurfaceSamples:
    points: np.ndarray
    normals: np.ndarray
    face_indices: np.ndarray


@dataclasses.dataclass(frozen=True)
class ContactPair:
    first: np.ndarray
    second: np.ndarray
    closing: np.ndarray
    width: float
    score: float

    @property
    def midpoint(self) -> np.ndarray:
        return (self.first + self.second) * 0.5


def _sample_faces(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if len(face_indices) == 0 or count <= 0:
        raise ValueError("face_indices and count must be non-empty")
    areas = np.asarray(mesh.area_faces[face_indices], dtype=np.float64)
    if not np.all(np.isfinite(areas)) or areas.sum() <= 0.0:
        raise ValueError("Mesh contains invalid face areas")
    selected = rng.choice(face_indices, size=count, p=areas / areas.sum())
    triangles = np.asarray(mesh.triangles[selected], dtype=np.float64)
    uv = rng.random((count, 2))
    reflected = uv.sum(axis=1) > 1.0
    uv[reflected] = 1.0 - uv[reflected]
    points = (
        triangles[:, 0]
        + uv[:, :1] * (triangles[:, 1] - triangles[:, 0])
        + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])
    )
    return points, selected


def sample_surface_mixed(
    mesh: trimesh.Trimesh,
    *,
    area_count: int,
    balanced_count: int,
    rng: np.random.Generator,
) -> SurfaceSamples:
    """Mix area sampling with component/normal-bin balanced sampling."""

    face_indices = np.arange(len(mesh.faces), dtype=np.int64)
    area_points, area_faces = _sample_faces(mesh, face_indices, area_count, rng)
    components = trimesh.graph.connected_component_labels(
        mesh.face_adjacency, node_count=len(mesh.faces)
    )
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    dominant = np.argmax(np.abs(normals), axis=1)
    signed_bins = dominant * 2 + (normals[np.arange(len(normals)), dominant] > 0)
    groups = sorted(set(zip(components.tolist(), signed_bins.tolist())))
    per_group = np.full(len(groups), balanced_count // len(groups), dtype=np.int64)
    per_group[: balanced_count % len(groups)] += 1
    balanced_points = []
    balanced_faces = []
    for group, count in zip(groups, per_group):
        if count == 0:
            continue
        mask = (components == group[0]) & (signed_bins == group[1])
        points, selected = _sample_faces(mesh, face_indices[mask], int(count), rng)
        balanced_points.append(points)
        balanced_faces.append(selected)
    all_points = np.concatenate([area_points, *balanced_points], axis=0)
    all_faces = np.concatenate([area_faces, *balanced_faces], axis=0)
    return SurfaceSamples(
        points=all_points,
        normals=normals[all_faces],
        face_indices=all_faces,
    )


def _pair_key(pair: ContactPair) -> tuple[float, float, float, float, float, float]:
    midpoint = pair.midpoint
    closing = pair.closing
    return (*np.round(midpoint, 9), *np.round(closing, 9))


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    result = np.asarray(axis, dtype=np.float64)
    for value in result:
        if abs(float(value)) > 1e-12:
            return result if value > 0.0 else -result
    raise ValueError("axis must be non-zero")


def _raycast_opposite_pairs(
    mesh: trimesh.Trimesh,
    samples: SurfaceSamples,
    config: AntipodalConfig,
) -> dict[int, ContactPair]:
    """Cast inward rays without the optional rtree dependency."""

    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    triangle_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    vertex0 = triangles[:, 0]
    edge1 = triangles[:, 1] - vertex0
    edge2 = triangles[:, 2] - vertex0
    cosine_limit = math.cos(math.radians(config.normal_error_degrees))
    ray_count = min(config.raycast_sample_limit, len(samples.points))
    resolved: dict[int, ContactPair] = {}
    for start in range(0, ray_count, config.raycast_batch_size):
        stop = min(start + config.raycast_batch_size, ray_count)
        first = samples.points[start:stop]
        first_normals = samples.normals[start:stop]
        directions = -first_normals
        origins = first + directions * 1e-6
        h = np.cross(directions[:, None, :], edge2[None, :, :])
        determinant = np.einsum("fj,bfj->bf", edge1, h)
        valid = np.abs(determinant) > 1e-12
        inverse = np.zeros_like(determinant)
        inverse[valid] = 1.0 / determinant[valid]
        displacement = origins[:, None, :] - vertex0[None, :, :]
        u = inverse * np.einsum("bfj,bfj->bf", displacement, h)
        q = np.cross(displacement, edge1[None, :, :])
        v = inverse * np.einsum("bj,bfj->bf", directions, q)
        distance = inverse * np.einsum("fj,bfj->bf", edge2, q)
        valid &= u >= -1e-9
        valid &= v >= -1e-9
        valid &= u + v <= 1.0 + 1e-9
        valid &= distance >= config.minimum_width
        valid &= distance <= config.maximum_width
        distance[~valid] = np.inf
        nearest_faces = np.argmin(distance, axis=1)
        nearest_distances = distance[np.arange(stop - start), nearest_faces]
        for offset, (face_index, width) in enumerate(
            zip(nearest_faces, nearest_distances)
        ):
            if not np.isfinite(width):
                continue
            sample_index = start + offset
            closing = directions[offset]
            second_normal = triangle_normals[face_index]
            first_alignment = float((-first_normals[offset]) @ closing)
            second_alignment = float(second_normal @ closing)
            if first_alignment < cosine_limit or second_alignment < cosine_limit:
                continue
            second = first[offset] + closing * width
            resolved[sample_index] = ContactPair(
                first=first[offset].copy(),
                second=second,
                closing=closing.copy(),
                width=float(width),
                score=0.5 * (first_alignment + second_alignment),
            )
    return resolved


def build_contact_pairs(
    samples: SurfaceSamples,
    config: AntipodalConfig,
    *,
    mesh: trimesh.Trimesh | None = None,
) -> list[ContactPair]:
    """Use radius queries and bounded neighbors; never enumerate all point pairs."""

    points = np.asarray(samples.points, dtype=np.float64)
    normals = np.asarray(samples.normals, dtype=np.float64)
    dominant = np.argmax(np.abs(normals), axis=1)
    positive = normals[np.arange(len(normals)), dominant] > 0.0
    normal_bins = dominant * 2 + positive
    bin_indices = {bin_id: np.flatnonzero(normal_bins == bin_id) for bin_id in range(6)}
    bin_trees = {
        bin_id: cKDTree(points[indices])
        for bin_id, indices in bin_indices.items()
        if len(indices)
    }
    cosine_limit = math.cos(math.radians(config.normal_error_degrees))
    ray_pairs = (
        _raycast_opposite_pairs(mesh, samples, config) if mesh is not None else {}
    )
    candidates: list[ContactPair] = list(ray_pairs.values())
    for first_index, (first, first_normal) in enumerate(zip(points, normals)):
        if first_index in ray_pairs:
            continue
        opposite_bin = int(dominant[first_index] * 2 + (not positive[first_index]))
        opposite_indices = bin_indices[opposite_bin]
        if len(opposite_indices) == 0:
            continue
        query_count = min(
            len(opposite_indices), max(32, config.maximum_opposites_per_point * 8)
        )
        distances, local_indices = bin_trees[opposite_bin].query(
            first,
            k=query_count,
            distance_upper_bound=config.maximum_width,
        )
        distances = np.atleast_1d(distances)
        local_indices = np.atleast_1d(local_indices)
        neighbor_indices = [
            int(opposite_indices[local_index])
            for distance, local_index in zip(distances, local_indices)
            if np.isfinite(distance) and local_index < len(opposite_indices)
        ]
        scored = []
        for second_index in neighbor_indices:
            if second_index <= first_index:
                continue
            delta = points[second_index] - first
            width = float(np.linalg.norm(delta))
            if width < config.minimum_width:
                continue
            closing = delta / width
            first_alignment = float((-first_normal) @ closing)
            second_alignment = float(normals[second_index] @ closing)
            if first_alignment < cosine_limit or second_alignment < cosine_limit:
                continue
            score = 0.5 * (first_alignment + second_alignment)
            scored.append((score, second_index, closing, width))
        scored.sort(key=lambda row: (-row[0], row[3], row[1]))
        for score, second_index, closing, width in scored[
            : config.maximum_opposites_per_point
        ]:
            candidates.append(
                ContactPair(
                    first=first.copy(),
                    second=points[second_index].copy(),
                    closing=closing.copy(),
                    width=width,
                    score=score,
                )
            )
    candidates.sort(key=lambda pair: (-pair.score, pair.width, _pair_key(pair)))
    kept: list[ContactPair] = []
    buckets: dict[tuple[int, ...], list[ContactPair]] = {}
    angle_limit = math.cos(math.radians(config.pair_axis_nms_degrees))
    axis_resolution = 2.0 * math.sin(math.radians(config.pair_axis_nms_degrees) * 0.5)
    for pair in candidates:
        midpoint_bin = tuple(
            int(value) for value in np.floor(pair.midpoint / config.pair_midpoint_nms)
        )
        axis_bin = tuple(
            int(value)
            for value in np.round(_canonical_axis(pair.closing) / axis_resolution)
        )
        bucket_key = (*midpoint_bin, *axis_bin)
        bucket = buckets.setdefault(bucket_key, [])
        if any(
            np.linalg.norm(pair.midpoint - existing.midpoint)
            <= config.pair_midpoint_nms
            and abs(float(pair.closing @ existing.closing)) >= angle_limit
            for existing in bucket
        ):
            continue
        kept.append(pair)
        bucket.append(pair)
        if len(kept) == config.maximum_contact_pairs:
            break
    return kept


def _orthogonal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(reference @ axis)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return first, second


def _rotation_distance_with_gripper_symmetry(
    first: np.ndarray, second: np.ndarray
) -> float:
    symmetry = np.diag([-1.0, -1.0, 1.0])

    def angle(delta: np.ndarray) -> float:
        return math.acos(float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)))

    return min(angle(first.T @ second), angle(first.T @ second @ symmetry))


def _pose_is_duplicate(
    transform: np.ndarray,
    nearby: list[GraspCandidate],
    config: AntipodalConfig,
) -> bool:
    angle_limit = math.radians(config.pose_rotation_nms_degrees)
    for candidate in nearby:
        existing = candidate.object_T_tcp
        if (
            np.linalg.norm(transform[:3, 3] - existing[:3, 3])
            <= config.pose_translation_nms
            and _rotation_distance_with_gripper_symmetry(
                transform[:3, :3], existing[:3, :3]
            )
            <= angle_limit
        ):
            return True
    return False


class AntipodalGraspProvider:
    def __init__(self, config: AntipodalConfig | None = None):
        self.config = config or AntipodalConfig()

    def generate(
        self, geometry: ResolvedObjectGeometry, *, seed: int
    ) -> list[GraspCandidate]:
        rng = np.random.default_rng(seed)
        mesh = geometry.proposal_mesh
        center_of_mass = np.asarray(
            mesh.center_mass if mesh.is_volume else mesh.centroid,
            dtype=np.float64,
        )
        if center_of_mass.shape != (3,) or not np.all(np.isfinite(center_of_mass)):
            raise ValueError("Proposal mesh has an invalid center of mass")
        samples = sample_surface_mixed(
            mesh,
            area_count=self.config.area_samples,
            balanced_count=self.config.balanced_samples,
            rng=rng,
        )
        pairs = build_contact_pairs(samples, self.config, mesh=mesh)
        proposals = []
        for pair_index, pair in enumerate(pairs):
            first, second = _orthogonal_basis(pair.closing)
            for roll_index in range(self.config.approaches_per_pair):
                angle = 2.0 * math.pi * roll_index / self.config.approaches_per_pair
                approach = math.cos(angle) * first + math.sin(angle) * second
                ortho = np.cross(pair.closing, approach)
                ortho /= np.linalg.norm(ortho)
                rotation = np.column_stack([ortho, pair.closing, approach])
                transform = np.eye(4)
                transform[:3, :3] = rotation
                transform[:3, 3] = pair.midpoint
                region = tuple(
                    int(value)
                    for value in np.floor(
                        pair.midpoint / self.config.contact_region_size
                    )
                )
                offset = center_of_mass - pair.midpoint
                com_distance = float(np.linalg.norm(offset))
                gravity_torque_risk = float(
                    np.linalg.norm(offset - pair.closing * float(offset @ pair.closing))
                )
                proposals.append(
                    (
                        -pair.score,
                        region,
                        pair_index,
                        roll_index,
                        com_distance,
                        gravity_torque_risk,
                        transform,
                        pair,
                    )
                )
        proposals.sort(key=lambda row: row[:4])
        selected: list[GraspCandidate] = []
        region_counts: dict[tuple[int, ...], int] = {}
        pose_bins: dict[tuple[int, int, int], list[GraspCandidate]] = {}
        for (
            _,
            region,
            pair_index,
            roll_index,
            com_distance,
            gravity_torque_risk,
            transform,
            pair,
        ) in proposals:
            if region_counts.get(region, 0) >= self.config.maximum_per_contact_region:
                continue
            position_bin = tuple(
                int(value)
                for value in np.floor(
                    transform[:3, 3] / self.config.pose_translation_nms
                )
            )
            nearby = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nearby.extend(
                            pose_bins.get(
                                (
                                    position_bin[0] + dx,
                                    position_bin[1] + dy,
                                    position_bin[2] + dz,
                                ),
                                (),
                            )
                        )
            if _pose_is_duplicate(transform, nearby, self.config):
                continue
            candidate = GraspCandidate(
                candidate_id=f"pair-{pair_index:04d}-roll-{roll_index:02d}",
                source=GraspProviderName.ANTIPODAL,
                object_T_tcp=transform,
                required_width=pair.width,
                proposal_score=pair.score,
                contact_points=np.stack([pair.first, pair.second]),
                metadata={
                    "provider_version": ANTIPODAL_PROVIDER_VERSION,
                    "contact_region": region,
                    "roll_index": roll_index,
                    "com_distance": com_distance,
                    "gravity_torque_risk": gravity_torque_risk,
                },
            )
            selected.append(candidate)
            pose_bins.setdefault(position_bin, []).append(candidate)
            region_counts[region] = region_counts.get(region, 0) + 1
            if len(selected) == self.config.maximum_candidates:
                break
        if not selected:
            raise RuntimeError(
                "proposal-empty: no antipodal contact pair passed filtering"
            )
        return selected
