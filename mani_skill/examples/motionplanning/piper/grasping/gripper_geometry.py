"""PiPER gripper collision geometry expressed in the ``piper_tcp`` frame."""

from __future__ import annotations

import dataclasses
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from transforms3d.euler import euler2mat

from mani_skill import PACKAGE_ASSET_DIR

PAD_DEPTH = 0.0015
PAD_COORDINATE_TOLERANCE = 2e-4


def _numbers(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(default if value is None else value.split(), dtype=np.float64)


def _origin_matrix(node: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if node is None:
        return transform
    transform[:3, :3] = euler2mat(*_numbers(node.attrib.get("rpy"), (0, 0, 0)))
    transform[:3, 3] = _numbers(node.attrib.get("xyz"), (0, 0, 0))
    return transform


@dataclasses.dataclass(frozen=True)
class PiperGripperGeometry:
    tcp_T_gripper_base: np.ndarray
    gripper_base_vertices: np.ndarray
    gripper_base_faces: np.ndarray
    finger_vertices: dict[str, np.ndarray]
    finger_faces: dict[str, np.ndarray]
    base_T_finger_origin: dict[str, np.ndarray]
    finger_axes: dict[str, np.ndarray]
    finger_bounds: dict[str, np.ndarray]

    @classmethod
    def from_package_assets(cls) -> PiperGripperGeometry:
        urdf_path = (
            pathlib.Path(PACKAGE_ASSET_DIR) / "robots/piper/piper_description.urdf"
        )
        root = ET.parse(urdf_path).getroot()
        links = {node.attrib["name"]: node for node in root.findall("link")}
        joints = {node.attrib["name"]: node for node in root.findall("joint")}
        base_T_tcp = _origin_matrix(joints["piper_tcp_joint"].find("origin"))
        tcp_T_base = np.linalg.inv(base_T_tcp)

        def collision_mesh(link_name: str) -> tuple[np.ndarray, np.ndarray]:
            collision = links[link_name].find("collision")
            if collision is None:
                raise ValueError(f"PIPER link has no collision geometry: {link_name}")
            mesh_node = collision.find("geometry/mesh")
            if mesh_node is None:
                raise ValueError(f"PIPER collision is not a mesh: {link_name}")
            path = urdf_path.parent / mesh_node.attrib["filename"]
            mesh = trimesh.load(path, force="mesh", process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            scale = _numbers(mesh_node.attrib.get("scale"), (1, 1, 1))
            vertices = vertices * scale
            origin = _origin_matrix(collision.find("origin"))
            return (
                (origin[:3, :3] @ vertices.T + origin[:3, 3:4]).T,
                faces,
            )

        gripper_base_vertices, gripper_base_faces = collision_mesh("gripper_base")
        finger_meshes = {name: collision_mesh(name) for name in ("link7", "link8")}
        finger_vertices = {name: value[0] for name, value in finger_meshes.items()}
        return cls(
            tcp_T_gripper_base=tcp_T_base,
            gripper_base_vertices=gripper_base_vertices,
            gripper_base_faces=gripper_base_faces,
            finger_vertices=finger_vertices,
            finger_faces={name: value[1] for name, value in finger_meshes.items()},
            base_T_finger_origin={
                name: _origin_matrix(joints[f"joint{name[-1]}"].find("origin"))
                for name in ("link7", "link8")
            },
            finger_axes={
                name: _numbers(
                    joints[f"joint{name[-1]}"].find("axis").attrib.get("xyz"),
                    (0, 0, 1),
                )
                for name in ("link7", "link8")
            },
            finger_bounds={
                name: np.stack([vertices.min(axis=0), vertices.max(axis=0)], axis=0)
                for name, vertices in finger_vertices.items()
            },
        )

    def tcp_link_transforms(self, width: float) -> dict[str, np.ndarray]:
        if not 0.0 <= width <= 0.07:
            raise ValueError("PIPER width must be in [0, 0.07] m")
        transforms = {"gripper_base": self.tcp_T_gripper_base}
        positions = {"link7": width * 0.5, "link8": -width * 0.5}
        for name in ("link7", "link8"):
            translation = np.eye(4)
            translation[:3, 3] = self.finger_axes[name] * positions[name]
            transforms[name] = (
                self.tcp_T_gripper_base
                @ self.base_T_finger_origin[name]
                @ translation
            )
        return transforms

    def link_meshes(self) -> dict[str, trimesh.Trimesh]:
        meshes = {
            "gripper_base": trimesh.Trimesh(
                vertices=self.gripper_base_vertices,
                faces=self.gripper_base_faces,
                process=False,
            )
        }
        meshes.update(
            {
                name: trimesh.Trimesh(
                    vertices=self.finger_vertices[name],
                    faces=self.finger_faces[name],
                    process=False,
                )
                for name in ("link7", "link8")
            }
        )
        return meshes

    def tcp_vertices(self, width: float) -> np.ndarray:
        if not 0.0 <= width <= 0.07:
            raise ValueError("PIPER width must be in [0, 0.07] m")
        transforms = self.tcp_link_transforms(width)
        result = []
        for name, vertices in {
            "gripper_base": self.gripper_base_vertices,
            **self.finger_vertices,
        }.items():
            transform = transforms[name]
            result.append(
                (transform[:3, :3] @ vertices.T + transform[:3, 3:4]).T
            )
        return np.concatenate(result, axis=0)

    def tcp_pad_vertices(self, link_name: str, *, width: float) -> np.ndarray:
        """Return the usable inner pad surface vertices in the TCP frame."""

        if link_name not in self.finger_vertices:
            raise ValueError(f"PIPER pad link must be link7 or link8: {link_name}")
        vertices = self.finger_vertices[link_name]
        faces = self.finger_faces[link_name]
        minimum_z = self.finger_bounds[link_name][0, 2]
        triangles = vertices[faces]
        pad_faces = np.all(
            triangles[:, :, 2] <= minimum_z + PAD_DEPTH + PAD_COORDINATE_TOLERANCE,
            axis=1,
        )
        if not np.any(pad_faces):
            raise RuntimeError(f"PIPER pad surface is empty: {link_name}")
        pad_vertices = np.unique(triangles[pad_faces].reshape(-1, 3), axis=0)
        transform = self.tcp_link_transforms(width)[link_name]
        return (
            transform[:3, :3] @ pad_vertices.T + transform[:3, 3:4]
        ).T

    def is_pad_point(
        self, link_name: str, link_T_world: np.ndarray, world_point: np.ndarray
    ) -> bool:
        if link_name not in self.finger_bounds:
            return False
        local = np.linalg.inv(link_T_world) @ np.append(world_point, 1.0)
        return self._is_pad_local_point(link_name, local[:3])

    def is_pad_contact(
        self,
        link_name: str,
        link_T_world: np.ndarray,
        world_point: np.ndarray,
        face_index: int,
    ) -> bool:
        """Classify an FCL contact by its actual gripper triangle.

        FCL's contact position is an intersection representative and is not
        guaranteed to lie on either colliding surface. Project it onto the
        reported gripper triangle before applying the pad-volume test.
        """

        if link_name not in self.finger_bounds:
            return False
        faces = self.finger_faces[link_name]
        if not 0 <= face_index < len(faces):
            raise ValueError(
                f"Invalid {link_name} collision face index: {face_index}"
            )
        world_T_link = np.asarray(link_T_world, dtype=np.float64)
        local_point = (
            np.linalg.inv(world_T_link)
            @ np.append(np.asarray(world_point, dtype=np.float64), 1.0)
        )[:3]
        triangle = self.finger_vertices[link_name][faces[face_index]]
        projected = trimesh.triangles.closest_point(
            triangle[None, :, :], local_point[None, :]
        )[0]
        return self._is_pad_local_point(link_name, projected)

    def _is_pad_local_point(self, link_name: str, local: np.ndarray) -> bool:
        bounds = self.finger_bounds[link_name]
        # The PiPER collision mesh's inner fingertip face is the local -Z
        # surface.  Local Y runs along the finger length; the previous
        # ``local[1] <= -0.02`` test excluded the actual fingertip pad, whose
        # planar contact surface is concentrated near Y=-1..-10 mm.
        return bool(
            bounds[0, 0] - PAD_COORDINATE_TOLERANCE
            <= local[0]
            <= bounds[1, 0] + PAD_COORDINATE_TOLERANCE
            and bounds[0, 1] - PAD_COORDINATE_TOLERANCE
            <= local[1]
            <= bounds[1, 1] + PAD_COORDINATE_TOLERANCE
            and bounds[0, 2] - PAD_COORDINATE_TOLERANCE
            <= local[2]
            <= bounds[0, 2] + PAD_DEPTH
        )
