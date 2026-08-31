"""Shared candidate transforms, filtering, and execution-time checks."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import sapien
import trimesh

from mani_skill.utils import sapien_utils

from .contracts import GraspCandidate
from .gripper_geometry import PiperGripperGeometry

PREGRASP_DISTANCE = 0.07
MAX_DESCEND_TRANSLATION_STEP = 0.002
LIFT_DISTANCE = 0.12
LEGACY_HOLD_STEPS = 3
ROBUST_HOLD_STEPS = 10


@dataclasses.dataclass(frozen=True)
class RankedCandidate:
    candidate: GraspCandidate
    clearance_score: float
    ik_cost: float
    path_length: float


def pose_matrix(pose: Any) -> np.ndarray:
    matrix = pose.to_transformation_matrix()
    if hasattr(matrix, "detach"):
        matrix = matrix.detach().cpu().numpy()
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (1, 4, 4):
        matrix = matrix[0]
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected one pose matrix, got {matrix.shape}")
    return matrix


def world_grasp_pose(object_pose: Any, candidate: GraspCandidate) -> sapien.Pose:
    return sapien.Pose(pose_matrix(object_pose) @ candidate.object_T_tcp)


def pregrasp_pose(grasp_pose: sapien.Pose) -> sapien.Pose:
    return grasp_pose * sapien.Pose([0.0, 0.0, -PREGRASP_DISTANCE])


def minimum_gripper_clearance(
    world_T_tcp: np.ndarray,
    geometry: PiperGripperGeometry,
    *,
    contact_width: float,
    table_height: float = 0.0,
    closure_samples: int = 8,
) -> float:
    if closure_samples < 2:
        raise ValueError("closure_samples must be at least two")
    minimum_z = math.inf
    for width in np.linspace(0.068, contact_width, closure_samples):
        points = geometry.tcp_vertices(float(width))
        world_points = (world_T_tcp[:3, :3] @ points.T + world_T_tcp[:3, 3:4]).T
        minimum_z = min(minimum_z, float(world_points[:, 2].min()))
    return minimum_z - table_height


def rank_candidates(
    candidates: Iterable[RankedCandidate],
    *,
    maximum_candidates: int = 16,
) -> list[RankedCandidate]:
    if maximum_candidates <= 0:
        raise ValueError("maximum_candidates must be positive")

    def bucket(value: float, width: float) -> int:
        return int(math.floor(value / width))

    ordered = sorted(
        candidates,
        key=lambda item: (
            -bucket(item.candidate.proposal_score, 0.05),
            -bucket(item.clearance_score, 0.005),
            item.ik_cost + item.path_length,
            item.candidate.candidate_id,
        ),
    )
    selected = []
    region_counts: dict[tuple[Any, ...], int] = {}
    approach_counts: dict[tuple[int, int, int], int] = {}
    closing_counts: dict[tuple[int, int, int], int] = {}
    selected_ids = set()
    for item in ordered:
        transform = item.candidate.object_T_tcp
        region = tuple(item.candidate.metadata.get("contact_region", (0, 0, 0)))
        approach = tuple(np.round(transform[:3, 2] * 2.0).astype(int))
        closing = tuple(np.round(np.abs(transform[:3, 1]) * 2.0).astype(int))
        if region_counts.get(region, 0) >= 4:
            continue
        if approach_counts.get(approach, 0) >= 6:
            continue
        if closing_counts.get(closing, 0) >= 8:
            continue
        selected.append(item)
        selected_ids.add(item.candidate.candidate_id)
        region_counts[region] = region_counts.get(region, 0) + 1
        approach_counts[approach] = approach_counts.get(approach, 0) + 1
        closing_counts[closing] = closing_counts.get(closing, 0) + 1
        if len(selected) == maximum_candidates:
            break
    if len(selected) < maximum_candidates:
        for item in ordered:
            if item.candidate.candidate_id in selected_ids:
                continue
            region = tuple(item.candidate.metadata.get("contact_region", (0, 0, 0)))
            if region_counts.get(region, 0) >= 4:
                continue
            selected.append(item)
            selected_ids.add(item.candidate.candidate_id)
            region_counts[region] = region_counts.get(region, 0) + 1
            if len(selected) == maximum_candidates:
                break
    return selected


def _sample_mesh_points(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    faces = rng.choice(len(mesh.faces), size=count, p=areas / areas.sum())
    triangles = np.asarray(mesh.triangles[faces])
    uv = rng.random((count, 2))
    reflected = uv.sum(axis=1) > 1.0
    uv[reflected] = 1.0 - uv[reflected]
    return (
        triangles[:, 0]
        + uv[:, :1] * (triangles[:, 1] - triangles[:, 0])
        + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])
    )


def scene_collision_points(base_env, *, include_target: bool) -> np.ndarray:
    grid_x = np.arange(-0.30, 0.301, 0.01)
    grid_y = np.arange(-0.30, 0.301, 0.01)
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="ij")
    table_points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    robot_base = np.asarray(base_env.agent.robot.pose.p[0].cpu(), dtype=np.float64)
    base_clearance = np.linalg.norm(table_points[:, :2] - robot_base[:2], axis=1)
    table_points = table_points[base_clearance >= 0.20]
    chunks = [table_points]
    actors = []
    if include_target:
        actors.append(base_env.obj)
    actors.extend(getattr(base_env, "_clutter_objs", ()))
    for index, actor in enumerate(actors):
        mesh = actor.get_first_collision_mesh(to_world_frame=True)
        if mesh is None:
            raise RuntimeError(f"scene-collision-mesh-missing: {actor.name}")
        chunks.append(_sample_mesh_points(mesh, 512, seed=index))
    return np.concatenate(chunks, axis=0)


def actor_collision_meshes(actor) -> tuple[trimesh.Trimesh, ...]:
    """Return the actor's actual PhysX collision shapes in actor-local space."""

    entity = actor._objs[0]
    body = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    if body is None:
        body = entity.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
    if body is None:
        raise RuntimeError(f"collision-body-missing: {actor.name}")
    meshes = []
    for shape in body.collision_shapes:
        if isinstance(
            shape,
            (
                sapien.physx.PhysxCollisionShapeConvexMesh,
                sapien.physx.PhysxCollisionShapeTriangleMesh,
            ),
        ):
            vertices = np.asarray(shape.vertices, dtype=np.float64) * np.asarray(
                shape.scale, dtype=np.float64
            )
            faces = np.asarray(shape.triangles, dtype=np.int64)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        elif isinstance(shape, sapien.physx.PhysxCollisionShapeBox):
            mesh = trimesh.creation.box(
                extents=2.0 * np.asarray(shape.half_size, dtype=np.float64)
            )
        else:
            raise TypeError(
                f"Unsupported target collision shape: {type(shape).__name__}"
            )
        mesh.apply_transform(pose_matrix(shape.local_pose))
        meshes.append(mesh)
    if not meshes:
        raise RuntimeError(f"collision-shapes-empty: {actor.name}")
    return tuple(meshes)


class TargetGeometryCollisionValidator:
    """Preflight gripper/target collisions against actual PhysX geometry."""

    def __init__(
        self,
        base_env,
        gripper_geometry: PiperGripperGeometry,
        *,
        penetration_tolerance: float = 0.002,
    ):
        try:
            manager = trimesh.collision.CollisionManager()
        except ValueError as error:
            raise RuntimeError(
                "Target geometry validation requires python-fcl; install "
                "requirements-lift.txt"
            ) from error
        self.base_env = base_env
        self.gripper_geometry = gripper_geometry
        self.gripper_meshes = gripper_geometry.link_meshes()
        self.penetration_tolerance = float(penetration_tolerance)
        self.manager = manager
        self.link_managers = {}
        for link_name, mesh in self.gripper_meshes.items():
            link_manager = trimesh.collision.CollisionManager()
            link_manager.add_object(link_name, mesh)
            self.link_managers[link_name] = link_manager
        self.target_names = []
        for index, mesh in enumerate(actor_collision_meshes(base_env.obj)):
            name = f"target-{index}"
            self.manager.add_object(name, mesh)
            self.target_names.append(name)

    def validate_pose(
        self,
        world_T_tcp: np.ndarray,
        *,
        width: float,
        allow_pad_contact: bool,
        progress: float,
    ) -> frozenset[str]:
        world_T_object = pose_matrix(self.base_env.obj.pose)
        contacting_fingers = set()
        for name in self.target_names:
            self.manager.set_transform(name, world_T_object)
        for link_name, tcp_T_link in self.gripper_geometry.tcp_link_transforms(
            width
        ).items():
            world_T_link = world_T_tcp @ tcp_T_link
            link_manager = self.link_managers[link_name]
            link_manager.set_transform(link_name, world_T_link)
            colliding, contacts = self.manager.in_collision_other(
                link_manager,
                return_data=True,
            )
            if not colliding:
                continue
            if link_name not in {"link7", "link8"}:
                raise RuntimeError(f"target-contact-{link_name}")
            if not allow_pad_contact or progress < 0.8:
                raise RuntimeError("target-pad-contact-early")
            if not contacts:
                raise RuntimeError(f"target-contact-unresolved-{link_name}")
            for contact in contacts:
                if float(contact.depth) > self.penetration_tolerance:
                    raise RuntimeError("target-penetration")
                if not self.gripper_geometry.is_pad_contact(
                    link_name,
                    world_T_link,
                    np.asarray(contact.point, dtype=np.float64),
                    int(contact.index(link_name)),
                ):
                    raise RuntimeError(f"target-non-pad-contact-{link_name}")
            contacting_fingers.add(link_name)
        return frozenset(contacting_fingers)

    def validate_descend_and_closure(
        self,
        pregrasp: sapien.Pose,
        grasp: sapien.Pose,
        *,
        contact_width: float,
        closure_samples: int | None = None,
    ) -> float:
        grasp_matrix = pose_matrix(grasp)
        waypoints = dense_translation_waypoints(pregrasp.p, grasp.p)
        for index, position in enumerate(waypoints):
            waypoint = grasp_matrix.copy()
            waypoint[:3, 3] = position
            self.validate_pose(
                waypoint,
                width=0.068,
                allow_pad_contact=True,
                progress=(index + 1) / len(waypoints),
            )
        if closure_samples is None:
            closure_samples = max(
                2, int(math.ceil((0.068 - contact_width) / 0.002)) + 1
            )
        elif closure_samples < 2:
            raise ValueError("closure_samples must be at least two")
        for index, width in enumerate(np.linspace(0.068, contact_width, closure_samples)):
            contacts = self.validate_pose(
                grasp_matrix,
                width=float(width),
                allow_pad_contact=True,
                # The 80% gate protects the free-space descend. Once closure
                # starts, pad contact is expected at any gripper width.
                progress=1.0,
            )
            if contacts == {"link7", "link8"}:
                return float(width)
        raise RuntimeError("target-bilateral-pad-contact-missing")


class TargetContactValidator:
    def __init__(
        self,
        base_env,
        gripper_geometry: PiperGripperGeometry,
        *,
        penetration_tolerance: float = 0.002,
    ):
        if base_env.gpu_sim_enabled:
            raise ValueError("target contact validation requires physx_cpu")
        self.base_env = base_env
        self.gripper_geometry = gripper_geometry
        self.penetration_tolerance = penetration_tolerance

    @staticmethod
    def _entity(value):
        return value._objs[0]

    def validate(self, *, allow_pad_contact: bool, progress: float) -> None:
        contacts = self.base_env.scene.get_contacts()
        target = self._entity(self.base_env.obj)
        fingers = {"link7", "link8"}
        for link in self.base_env.agent.robot.get_links():
            pairwise = sapien_utils.get_pairwise_contacts(
                contacts, self._entity(link), target
            )
            for contact, _ in pairwise:
                for point in contact.points:
                    if point.separation < -self.penetration_tolerance:
                        raise RuntimeError("target-penetration")
                    if link.name not in fingers:
                        raise RuntimeError(f"target-contact-{link.name}")
                    if not allow_pad_contact or progress < 0.8:
                        raise RuntimeError("target-pad-contact-early")
                    if not self.gripper_geometry.is_pad_point(
                        link.name,
                        pose_matrix(link.pose),
                        np.asarray(point.position),
                    ):
                        raise RuntimeError(f"target-non-pad-contact-{link.name}")


def dense_translation_waypoints(
    start: np.ndarray,
    end: np.ndarray,
    *,
    maximum_step: float = MAX_DESCEND_TRANSLATION_STEP,
) -> np.ndarray:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    distance = float(np.linalg.norm(end - start))
    steps = max(1, int(math.ceil(distance / maximum_step)))
    return np.linspace(start, end, steps + 1)[1:]
