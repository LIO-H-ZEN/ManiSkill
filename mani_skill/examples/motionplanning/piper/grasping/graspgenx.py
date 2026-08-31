"""Import offline GraspGen-X proposals into the PiPER grasp contract."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import trimesh
import yaml

from .antipodal import sample_surface_mixed
from .contracts import GraspCandidate, GraspProviderName
from .geometry import ResolvedObjectGeometry
from .gripper_geometry import PiperGripperGeometry

GRASPGENX_PROVIDER_VERSION = "graspgenx_complete_mesh_v2_contact_refined"
GRASPGENX_IMPLEMENTATION_REVISION = "b9429097728cb1c430dd78b92edf17ba318aad03"
GRASPGENX_CHECKPOINT_REVISION = "7c834043c11a11417e31d6d5ea9355801e40a2c1"
GRASPGENX_GRIPPER_ASSETS_REVISION = "19a03c00d19aeaf052d0f6801f0041982d676e8a"
GRASPGENX_GRIPPER_NAME = "piper_hand"
GRASPGENX_GENERATION_CONTRACT_VERSION = "graspgenx_complete_mesh_generation_v1"


def _sha256_array(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())


@dataclasses.dataclass(frozen=True)
class GraspGenXConfig:
    """Versioned frame and model contract for offline GraspGen-X output."""

    implementation_revision: str = GRASPGENX_IMPLEMENTATION_REVISION
    checkpoint_revision: str = GRASPGENX_CHECKPOINT_REVISION
    gripper_assets_revision: str = GRASPGENX_GRIPPER_ASSETS_REVISION
    gripper_name: str = GRASPGENX_GRIPPER_NAME
    piper_tcp_offset: float = 0.1358
    minimum_contact_width: float = 0.001
    maximum_candidates: int = 256
    generation_seed: int = 0
    generation_num_grasps: int = 1024
    generation_topk_num_grasps: int = 256
    generation_num_sample_points: int = 3500
    generation_grasp_threshold: float = -1.0
    area_samples: int = 3072
    balanced_samples: int = 1024
    minimum_normal_alignment: float = 0.7
    pad_edge_margin: float = 0.001
    contact_compression: float = 0.0005
    maximum_closing_axis_correction: float = 0.012

    def __post_init__(self) -> None:
        for field_name in (
            "implementation_revision",
            "checkpoint_revision",
            "gripper_assets_revision",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        if self.gripper_name != GRASPGENX_GRIPPER_NAME:
            raise ValueError("GraspGen-X provider currently supports only piper_hand")
        if not np.isclose(self.piper_tcp_offset, 0.1358, atol=1e-8):
            raise ValueError("piper_tcp_offset must match the ManiSkill PiPER URDF")
        if not np.isclose(self.minimum_contact_width, 0.001, atol=1e-8):
            raise ValueError("minimum_contact_width must match the grasp contract")
        if self.maximum_candidates <= 0:
            raise ValueError("maximum_candidates must be positive")
        if self.generation_seed < 0:
            raise ValueError("generation_seed must be non-negative")
        if self.generation_num_grasps < self.generation_topk_num_grasps:
            raise ValueError(
                "generation_num_grasps must be at least generation_topk_num_grasps"
            )
        if self.generation_topk_num_grasps != self.maximum_candidates:
            raise ValueError(
                "generation_topk_num_grasps must equal maximum_candidates"
            )
        if self.generation_num_sample_points <= 0:
            raise ValueError("generation_num_sample_points must be positive")
        if not np.isclose(self.generation_grasp_threshold, -1.0, atol=1e-8):
            raise ValueError(
                "generation_grasp_threshold must be -1.0 for deterministic top-k"
            )
        if self.area_samples <= 0 or self.balanced_samples <= 0:
            raise ValueError("surface sample counts must be positive")
        if not 0.0 < self.minimum_normal_alignment <= 1.0:
            raise ValueError("minimum_normal_alignment must be in (0, 1]")
        if not 0.0 <= self.pad_edge_margin < 0.01:
            raise ValueError("pad_edge_margin must be in [0, 0.01) m")
        if not 0.0 < self.contact_compression <= 0.002:
            raise ValueError("contact_compression must be in (0, 0.002] m")
        if not 0.0 < self.maximum_closing_axis_correction <= 0.02:
            raise ValueError(
                "maximum_closing_axis_correction must be in (0, 0.02] m"
            )

    @property
    def hand_T_tcp(self) -> np.ndarray:
        # The official piper_hand URDF fixes gripper_base at Rz(+90 degrees)
        # in the model hand frame. Its fingertips end at z=0.1358 m. Applying
        # that same transform yields the exact ManiSkill piper_tcp convention:
        # columns [ortho, closing, approach].
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        transform[:3, 3] = [0.0, 0.0, self.piper_tcp_offset]
        return transform

    @property
    def fingerprint(self) -> str:
        payload = dataclasses.asdict(self)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class GraspGenXGraspProvider:
    """Convert model-scored hand poses; execution validity remains downstream."""

    def __init__(self, config: GraspGenXConfig | None = None):
        self.config = config or GraspGenXConfig()

    def cache_fingerprint(self, geometry: ResolvedObjectGeometry) -> str:
        digest = hashlib.sha256()
        digest.update(self.config.fingerprint.encode())
        digest.update(geometry.canonical_geometry_hash.encode())
        for index, mesh in enumerate(geometry.collision_meshes):
            _sha256_array(
                digest,
                f"collision_vertices_{index}",
                np.asarray(mesh.vertices, dtype="<f8"),
            )
            _sha256_array(
                digest,
                f"collision_faces_{index}",
                np.asarray(mesh.faces, dtype="<i8"),
            )
        return digest.hexdigest()

    def validate_generation_manifest(self, grasp_yaml: pathlib.Path) -> dict[str, Any]:
        input_path = pathlib.Path(grasp_yaml)
        manifest_path = input_path.with_suffix(input_path.suffix + ".generation.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "missing deterministic GraspGen-X generation manifest: "
                f"{manifest_path}"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("GraspGen-X generation manifest must be a JSON mapping")
        expected = {
            "contract_version": GRASPGENX_GENERATION_CONTRACT_VERSION,
            "implementation_revision": self.config.implementation_revision,
            "checkpoint_revision": self.config.checkpoint_revision,
            "gripper_assets_revision": self.config.gripper_assets_revision,
            "gripper_name": self.config.gripper_name,
            "seed": self.config.generation_seed,
            "num_grasps": self.config.generation_num_grasps,
            "topk_num_grasps": self.config.generation_topk_num_grasps,
            "num_sample_points": self.config.generation_num_sample_points,
            "grasp_threshold": self.config.generation_grasp_threshold,
        }
        for name, expected_value in expected.items():
            if payload.get(name) != expected_value:
                raise ValueError(
                    "GraspGen-X generation manifest mismatch for "
                    f"{name}: {payload.get(name)!r} != {expected_value!r}"
                )
        actual_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if payload.get("output_sha256") != actual_hash:
            raise ValueError("GraspGen-X YAML SHA-256 does not match its manifest")
        return dict(payload)

    def refine_contact_candidates(
        self,
        candidates: list[GraspCandidate],
        geometry: ResolvedObjectGeometry,
        gripper_geometry: PiperGripperGeometry,
    ) -> list[GraspCandidate]:
        """Center raw model poses on two physical PiPER pad contact surfaces."""

        if not geometry.collision_meshes:
            raise ValueError("GraspGen-X refinement requires collision geometry")
        sampling_mesh = trimesh.util.concatenate(
            [mesh.copy() for mesh in geometry.collision_meshes]
        )
        if not isinstance(sampling_mesh, trimesh.Trimesh) or not len(
            sampling_mesh.faces
        ):
            raise ValueError("GraspGen-X collision geometry is empty")
        seed_payload = (
            geometry.canonical_geometry_hash + self.config.fingerprint
        ).encode()
        seed = int(hashlib.sha256(seed_payload).hexdigest()[:16], 16)
        samples = sample_surface_mixed(
            sampling_mesh,
            area_count=self.config.area_samples,
            balanced_count=self.config.balanced_samples,
            rng=np.random.default_rng(seed),
        )

        pad_vertices = {
            name: gripper_geometry.tcp_pad_vertices(name, width=0.0)
            for name in ("link7", "link8")
        }
        x_min = max(points[:, 0].min() for points in pad_vertices.values())
        x_max = min(points[:, 0].max() for points in pad_vertices.values())
        z_min = max(points[:, 2].min() for points in pad_vertices.values())
        z_max = min(points[:, 2].max() for points in pad_vertices.values())
        margin = self.config.pad_edge_margin
        if x_min + margin >= x_max - margin or z_min + margin >= z_max - margin:
            raise RuntimeError("PiPER pad footprint is smaller than pad_edge_margin")
        closed_pad_overlap = abs(
            float(pad_vertices["link7"][:, 1].mean())
            - float(pad_vertices["link8"][:, 1].mean())
        )

        refined = []
        for candidate in candidates:
            tcp_T_object = np.linalg.inv(candidate.object_T_tcp)
            tcp_points = (
                tcp_T_object[:3, :3] @ samples.points.T
                + tcp_T_object[:3, 3:4]
            ).T
            tcp_normals = tcp_T_object[:3, :3] @ samples.normals.T
            footprint = (
                (tcp_points[:, 0] >= x_min + margin)
                & (tcp_points[:, 0] <= x_max - margin)
                & (tcp_points[:, 2] >= z_min + margin)
                & (tcp_points[:, 2] <= z_max - margin)
            )
            negative = np.flatnonzero(
                footprint
                & (tcp_normals[1] <= -self.config.minimum_normal_alignment)
            )
            positive = np.flatnonzero(
                footprint
                & (tcp_normals[1] >= self.config.minimum_normal_alignment)
            )
            if not len(negative) or not len(positive):
                continue
            negative_index = int(negative[np.argmin(tcp_points[negative, 1])])
            positive_index = int(positive[np.argmax(tcp_points[positive, 1])])
            negative_y = float(tcp_points[negative_index, 1])
            positive_y = float(tcp_points[positive_index, 1])
            local_width = positive_y - negative_y
            if local_width <= 0.0:
                continue
            closing_offset = 0.5 * (negative_y + positive_y)
            if (
                abs(closing_offset)
                > self.config.maximum_closing_axis_correction
            ):
                continue
            required_width = (
                local_width
                + closed_pad_overlap
                - self.config.contact_compression
            )
            if not self.config.minimum_contact_width <= required_width <= 0.068:
                continue
            object_T_tcp = candidate.object_T_tcp.copy()
            object_T_tcp[:3, 3] += (
                candidate.object_T_tcp[:3, 1] * closing_offset
            )
            contact_indices = np.asarray(
                [negative_index, positive_index], dtype=np.int64
            )
            contact_points = samples.points[contact_indices]
            contact_center = contact_points.mean(axis=0)
            refined.append(
                dataclasses.replace(
                    candidate,
                    object_T_tcp=object_T_tcp,
                    required_width=float(required_width),
                    contact_points=contact_points,
                    metadata={
                        **candidate.metadata,
                        "raw_object_T_tcp": candidate.object_T_tcp.tolist(),
                        "closing_axis_correction": closing_offset,
                        "estimated_contact_width": local_width,
                        "closed_pad_overlap": closed_pad_overlap,
                        "contact_compression": self.config.contact_compression,
                        "contact_region": tuple(
                            int(item) for item in np.floor(contact_center / 0.015)
                        ),
                        "contact_surface": geometry.collision_source,
                        "contact_refined": True,
                    },
                )
            )
        if not refined:
            raise RuntimeError(
                "proposal-empty: no GraspGen-X pose had opposing surfaces "
                "inside the PiPER pad footprint"
            )
        return refined

    def load_isaac_yaml(
        self,
        path: pathlib.Path,
        geometry: ResolvedObjectGeometry,
    ) -> list[GraspCandidate]:
        input_path = pathlib.Path(path)
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        payload = yaml.safe_load(input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("GraspGen-X output must be a YAML mapping")
        if (
            payload.get("format") != "isaac_grasp"
            or payload.get("format_version") != 1.0
        ):
            raise ValueError("Unsupported GraspGen-X output format")
        rows = payload.get("grasps")
        if not isinstance(rows, Mapping) or not rows:
            raise ValueError("GraspGen-X output contains no grasps")
        if len(rows) != self.config.generation_topk_num_grasps:
            raise ValueError(
                "GraspGen-X output candidate count does not match the generation "
                "contract: "
                f"{len(rows)} != {self.config.generation_topk_num_grasps}"
            )

        parsed = []
        for source_id, value in rows.items():
            if not isinstance(value, Mapping):
                raise TypeError(f"Invalid GraspGen-X row: {source_id}")
            position = np.asarray(value.get("position"), dtype=np.float64)
            orientation = value.get("orientation")
            if not isinstance(orientation, Mapping):
                raise TypeError(f"Missing orientation for {source_id}")
            quaternion = np.concatenate(
                [
                    np.asarray([orientation.get("w")], dtype=np.float64),
                    np.asarray(orientation.get("xyz"), dtype=np.float64),
                ]
            )
            score = float(value.get("confidence"))
            if (
                position.shape != (3,)
                or quaternion.shape != (4,)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(quaternion))
                or not np.isfinite(score)
            ):
                raise ValueError(f"Non-finite or malformed grasp: {source_id}")
            norm = float(np.linalg.norm(quaternion))
            if norm <= 1e-8:
                raise ValueError(f"Zero quaternion for {source_id}")
            quaternion /= norm
            w, x, y, z = quaternion
            object_T_hand = np.eye(4, dtype=np.float64)
            object_T_hand[:3, :3] = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ],
                dtype=np.float64,
            )
            object_T_hand[:3, 3] = position
            object_T_tcp = (
                geometry.object_T_mesh @ object_T_hand @ self.config.hand_T_tcp
            )
            contact_center = (
                geometry.object_T_mesh
                @ object_T_hand
                @ np.array([0.0, 0.0, 0.105, 1.0], dtype=np.float64)
            )
            closing = object_T_tcp[:3, 1]
            contacts = np.stack(
                [
                    contact_center[:3]
                    - closing * self.config.minimum_contact_width * 0.5,
                    contact_center[:3]
                    + closing * self.config.minimum_contact_width * 0.5,
                ]
            )
            parsed.append(
                (
                    score,
                    str(source_id),
                    object_T_hand,
                    object_T_tcp,
                    contacts,
                    contact_center[:3],
                )
            )

        parsed.sort(key=lambda item: (-item[0], item[1]))
        return [
            GraspCandidate(
                candidate_id=f"graspgenx-{rank:04d}",
                source=GraspProviderName.GRASPGENX,
                object_T_tcp=object_T_tcp,
                required_width=self.config.minimum_contact_width,
                proposal_score=score,
                contact_points=contacts,
                metadata={
                    "source_candidate_id": source_id,
                    "raw_object_T_hand": object_T_hand.tolist(),
                    "model_score": score,
                    "contact_region": tuple(
                        int(item) for item in np.floor(contact_center / 0.015)
                    ),
                    "pair_index": rank,
                    "provider_version": GRASPGENX_PROVIDER_VERSION,
                    "implementation_revision": self.config.implementation_revision,
                    "checkpoint_revision": self.config.checkpoint_revision,
                    "gripper_assets_revision": self.config.gripper_assets_revision,
                    "gripper_name": self.config.gripper_name,
                },
            )
            for rank, (
                score,
                source_id,
                object_T_hand,
                object_T_tcp,
                contacts,
                contact_center,
            ) in enumerate(parsed)
        ]
