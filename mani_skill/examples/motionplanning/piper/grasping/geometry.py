"""Resolve task objects into canonical actor-local proposal geometry."""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from typing import Literal

import numpy as np
import trimesh

from mani_skill import ASSET_DIR
from mani_skill.envs.tasks.pick_anything.episode_specs import ObjectSpec
from mani_skill.envs.tasks.pick_anything.randomization.object_sources import (
    InternDataAssetsSource,
)
from mani_skill.utils.io_utils import load_json

GEOMETRY_PREPROCESS_VERSION = "resolved_object_geometry_v1"


@dataclasses.dataclass(frozen=True)
class ResolvedObjectGeometry:
    object_spec: ObjectSpec
    proposal_mesh: trimesh.Trimesh
    collision_meshes: tuple[trimesh.Trimesh, ...]
    object_T_mesh: np.ndarray
    source_files: tuple[pathlib.Path, ...]
    collision_source: Literal["primitive", "mesh", "sapien_vhacd"]
    canonical_geometry_hash: str


def _load_mesh(path: pathlib.Path, scale: float) -> trimesh.Trimesh:
    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Expected one mesh in {path}, got {type(loaded).__name__}")
    mesh = loaded.copy()
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(scale)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no geometry: {path}")
    if not np.all(np.isfinite(mesh.vertices)):
        raise ValueError(f"Mesh contains non-finite vertices: {path}")
    mesh.remove_unreferenced_vertices()
    return mesh


def _canonical_geometry_hash(mesh: trimesh.Trimesh) -> str:
    vertices = np.ascontiguousarray(np.asarray(mesh.vertices, dtype="<f8"))
    faces = np.ascontiguousarray(np.asarray(mesh.faces, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(GEOMETRY_PREPROCESS_VERSION.encode())
    digest.update(np.asarray(vertices.shape, dtype="<i8").tobytes())
    digest.update(vertices.tobytes())
    digest.update(np.asarray(faces.shape, dtype="<i8").tobytes())
    digest.update(faces.tobytes())
    return digest.hexdigest()


def _resolve_ycb(spec: ObjectSpec) -> ResolvedObjectGeometry:
    metadata_path = ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"YCB metadata is missing at {metadata_path}; download the ycb asset first"
        )
    metadata = load_json(metadata_path)
    if spec.object_id not in metadata:
        raise ValueError(f"Unknown YCB object ID: {spec.object_id}")
    scale = float(metadata[spec.object_id].get("scales", [1.0])[0])
    model_dir = ASSET_DIR / "assets/mani_skill2_ycb/models" / spec.object_id
    visual_path = model_dir / "textured.obj"
    collision_path = model_dir / "collision.ply"
    proposal = _load_mesh(visual_path, scale)
    collision = _load_mesh(collision_path, scale)
    return ResolvedObjectGeometry(
        object_spec=spec,
        proposal_mesh=proposal,
        collision_meshes=(collision,),
        object_T_mesh=np.eye(4),
        source_files=(metadata_path, visual_path, collision_path),
        collision_source="mesh",
        canonical_geometry_hash=_canonical_geometry_hash(proposal),
    )


def _resolve_interndata(spec: ObjectSpec) -> ResolvedObjectGeometry:
    source = InternDataAssetsSource(categories=[str(spec.category)])
    obj_path = pathlib.Path(
        source._download_instance(str(spec.category), spec.object_id)
    )
    scale = source._resolve_scale(str(obj_path))
    proposal = _load_mesh(obj_path, scale)
    return ResolvedObjectGeometry(
        object_spec=spec,
        proposal_mesh=proposal,
        collision_meshes=(proposal.copy(),),
        object_T_mesh=np.eye(4),
        source_files=(obj_path,),
        collision_source="sapien_vhacd",
        canonical_geometry_hash=_canonical_geometry_hash(proposal),
    )


def resolve_object_geometry(spec: ObjectSpec) -> ResolvedObjectGeometry:
    """Resolve the same scale and local frame used by the task actor builder."""

    if spec.source == "cube":
        half_size = float(spec.cube_half_size)
        mesh = trimesh.creation.box(extents=[2.0 * half_size] * 3)
        return ResolvedObjectGeometry(
            object_spec=spec,
            proposal_mesh=mesh,
            collision_meshes=(mesh.copy(),),
            object_T_mesh=np.eye(4),
            source_files=(),
            collision_source="primitive",
            canonical_geometry_hash=_canonical_geometry_hash(mesh),
        )
    if spec.source == "ycb":
        return _resolve_ycb(spec)
    if spec.source == "interndata":
        return _resolve_interndata(spec)
    raise ValueError(f"Unsupported object source: {spec.source}")
