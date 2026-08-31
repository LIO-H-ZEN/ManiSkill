import copy
import fcntl
import hashlib
import os
import pathlib
import tempfile
import xml.etree.ElementTree as ET

import mplib
import numpy as np
import sapien
import trimesh

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.two_finger_gripper.motionplanner import (
    TwoFingerGripperMotionPlanningSolver,
)


class PiperMotionPlanningSolver(TwoFingerGripperMotionPlanningSolver):
    """Single-environment PIPER solver that stops at episode boundaries."""

    OPEN = 1.0
    CLOSED = -1.0
    MOVE_GROUP = "piper_tcp"
    ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
    SUPPORTED_CONTROL_MODES = frozenset(
        {"pd_joint_pos", "robodojo_pd_joint_pos"}
    )
    COLLISION_PROXY_VERSION = "piper_collision_proxy_v4_coacd"
    COACD_PARAMETERS = {
        "threshold": 0.08,
        "max_convex_hull": 8,
        "preprocess_mode": "on",
        "preprocess_resolution": 20,
        "resolution": 1000,
        "mcts_nodes": 10,
        "mcts_iterations": 50,
        "mcts_max_depth": 3,
        "seed": 0,
    }

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = False,
        base_pose: sapien.Pose | None = None,
        visualize_target_grasp_pose: bool = False,
        print_env_info: bool = False,
        joint_vel_limits: float = 0.9,
        joint_acc_limits: float = 0.9,
    ):
        if env.unwrapped.num_envs != 1:
            raise ValueError("PIPER MPLib expert requires num_envs=1")
        if env.unwrapped.control_mode not in self.SUPPORTED_CONTROL_MODES:
            raise ValueError(
                "PIPER MPLib expert requires one of "
                f"{sorted(self.SUPPORTED_CONTROL_MODES)}, got "
                f"{env.unwrapped.control_mode!r}"
            )
        super().__init__(
            env,
            debug,
            vis,
            base_pose,
            visualize_target_grasp_pose,
            print_env_info,
            joint_vel_limits,
            joint_acc_limits,
        )
        self.episode_done = False
        self.last_transition = None
        self.waypoint_validator = None
        self.execution_stage = "idle"

    @classmethod
    def _build_collision_planning_urdf(
        cls, source_path: str
    ) -> pathlib.Path:
        source = pathlib.Path(source_path)
        source_bytes = source.read_bytes()
        root = ET.fromstring(source_bytes)
        srdf_path = source.with_suffix(".srdf")
        if not srdf_path.is_file():
            raise FileNotFoundError(f"PIPER SRDF is missing: {srdf_path}")
        dependencies = [source_bytes, srdf_path.read_bytes()]
        for mesh_node in root.findall(".//collision/geometry/mesh"):
            filename = mesh_node.attrib.get("filename")
            if not filename:
                raise ValueError("PIPER collision mesh is missing filename")
            path = pathlib.Path(filename)
            if not path.is_absolute():
                path = source.parent / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            dependencies.append(path.read_bytes())
        digest_builder = hashlib.sha256(cls.COLLISION_PROXY_VERSION.encode())
        for contents in dependencies:
            digest_builder.update(len(contents).to_bytes(8, "little"))
            digest_builder.update(contents)
        digest = digest_builder.hexdigest()
        output_dir = pathlib.Path(tempfile.gettempdir()) / "maniskill_piper_mplib"
        cache_dir = output_dir / digest
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / "piper_collision.urdf"
        lock_path = cache_dir / ".build.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if output_path.exists():
                return output_path

            try:
                import coacd
            except ModuleNotFoundError as error:
                raise ModuleNotFoundError(
                    "PiPER collision-aware planning requires coacd; install "
                    "requirements-lift.txt"
                ) from error
            coacd.set_log_level("off")

            for link in root.findall("link"):
                for tag in ("visual", "inertial"):
                    for node in list(link.findall(tag)):
                        link.remove(node)
                collisions = list(link.findall("collision"))
                for collision_index, collision in enumerate(collisions):
                    mesh_node = collision.find("geometry/mesh")
                    if mesh_node is None:
                        continue
                    path = pathlib.Path(mesh_node.attrib["filename"])
                    if not path.is_absolute():
                        path = source.parent / path
                    path = path.resolve()
                    mesh = trimesh.load(path, force="mesh", process=True)
                    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                        raise ValueError(f"Invalid PIPER collision mesh: {path}")
                    parts = coacd.run_coacd(
                        coacd.Mesh(
                            np.asarray(mesh.vertices, dtype=np.float64),
                            np.asarray(mesh.faces, dtype=np.int32),
                        ),
                        **cls.COACD_PARAMETERS,
                    )
                    if len(parts) < 1:
                        raise RuntimeError(
                            f"CoACD returned no PiPER collision parts for {path}"
                        )
                    link.remove(collision)
                    for part_index, (vertices, faces) in enumerate(parts):
                        part = trimesh.Trimesh(
                            vertices=np.asarray(vertices, dtype=np.float64),
                            faces=np.asarray(faces, dtype=np.int64),
                            process=True,
                        )
                        if not part.is_watertight or not part.is_volume:
                            raise RuntimeError(
                                f"CoACD produced an invalid collision part for {path}"
                            )
                        proxy_path = cache_dir / (
                            f"{link.attrib['name']}-{collision_index}-{part_index}.convex.stl"
                        )
                        temporary_proxy = proxy_path.with_suffix(
                            f".{os.getpid()}.tmp"
                        )
                        part.export(temporary_proxy, file_type="stl")
                        os.replace(temporary_proxy, proxy_path)
                        proxy_collision = copy.deepcopy(collision)
                        proxy_mesh = proxy_collision.find("geometry/mesh")
                        if proxy_mesh is None:
                            raise RuntimeError("Copied collision lost its mesh geometry")
                        proxy_mesh.set("filename", proxy_path.name)
                        link.append(proxy_collision)

            for mesh_node in root.findall(".//collision/geometry/mesh"):
                cached_mesh = cache_dir / mesh_node.attrib["filename"]
                if not cached_mesh.is_file():
                    raise RuntimeError(
                        f"Planning collision mesh is missing: {cached_mesh}"
                    )
            contents = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            temporary_path = output_path.with_suffix(f".{os.getpid()}.tmp")
            temporary_path.write_bytes(contents)
            os.replace(temporary_path, output_path)
        return output_path

    _build_kinematic_planning_urdf = _build_collision_planning_urdf

    def setup_planner(self):
        planning_urdf = self._build_collision_planning_urdf(
            str(self.env_agent.urdf_path)
        )
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=str(planning_urdf),
            srdf=str(self.env_agent.urdf_path).replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=self.MOVE_GROUP,
        )
        planner.joint_vel_limits = (
            np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
        )
        planner.joint_acc_limits = (
            np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
        )
        return planner

    def set_collision_point_cloud(
        self, points: np.ndarray, *, radius: float = 0.005
    ) -> None:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("collision point cloud must have shape (N, 3)")
        if len(points) == 0 or radius <= 0.0:
            raise ValueError("collision point cloud and radius must be non-empty")
        base_matrix = self.base_pose.inv().to_transformation_matrix()
        self.all_collision_pts = (
            base_matrix[:3, :3] @ points.T + base_matrix[:3, 3:4]
        ).T
        self.use_point_cloud = True
        self.planner.update_point_cloud(self.all_collision_pts, radius=radius)

    def _transform_pose_for_planning(self, target: sapien.Pose) -> sapien.Pose:
        # MPLib 0.1.1 fails to convert world targets for a non-zero base pose.
        # Keep its planning base at identity and provide explicit base-frame poses.
        return self.base_pose.inv() * target

    def plan_screw_from_qpos(
        self, target: sapien.Pose, start_qpos: np.ndarray
    ) -> dict:
        """Plan a Cartesian screw motion from an explicit robot state."""

        start_qpos = np.asarray(start_qpos, dtype=np.float64)
        expected_shape = tuple(self.robot.get_qpos().shape[1:])
        if start_qpos.shape != expected_shape or not np.all(np.isfinite(start_qpos)):
            raise ValueError(
                f"start_qpos must have shape {expected_shape} and be finite"
            )
        planning_target = self._transform_pose_for_planning(target)
        payload = np.concatenate([planning_target.p, planning_target.q])
        result = None
        for _ in range(2):
            result = self.planner.plan_screw(
                payload,
                start_qpos,
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            if result["status"] == "Success":
                return result
        if result is None:
            raise RuntimeError("MPlib screw planner did not run")
        return result

    @staticmethod
    def _as_bool(value) -> bool:
        return bool(np.asarray(value).reshape(-1)[0])

    def validate_arm_path(self, result: dict, *, stage: str) -> np.ndarray:
        """Reject planner paths that the environment action space would clip."""

        if not stage:
            raise ValueError("path validation stage must not be empty")
        positions = np.asarray(result["position"], dtype=np.float32)
        if (
            positions.ndim != 2
            or positions.shape[1] != len(self.ARM_JOINT_NAMES)
            or len(positions) == 0
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError(f"Invalid PIPER path shape or values: {positions.shape}")
        qlimits = self.robot.get_qlimits()
        if hasattr(qlimits, "detach"):
            qlimits = qlimits.detach().cpu().numpy()
        qlimits = np.asarray(qlimits, dtype=np.float32)
        if qlimits.ndim != 3 or qlimits.shape[0] != 1 or qlimits.shape[2] != 2:
            raise ValueError(f"Invalid PIPER joint-limit shape: {qlimits.shape}")
        arm_limits = qlimits[0, : len(self.ARM_JOINT_NAMES)]
        if arm_limits.shape != (len(self.ARM_JOINT_NAMES), 2):
            raise ValueError(f"Invalid PIPER arm joint-limit shape: {arm_limits.shape}")
        violations = (positions < arm_limits[:, 0]) | (
            positions > arm_limits[:, 1]
        )
        if np.any(violations):
            waypoint_index, joint_index = np.argwhere(violations)[0]
            value = float(positions[waypoint_index, joint_index])
            lower, upper = (float(item) for item in arm_limits[joint_index])
            raise RuntimeError(
                f"{stage}: joint-limit-{self.ARM_JOINT_NAMES[joint_index]} "
                f"at waypoint {waypoint_index}: {value:.9f} not in "
                f"[{lower:.9f}, {upper:.9f}]"
            )
        return positions

    def _step(self, action: np.ndarray):
        if self.episode_done:
            raise RuntimeError("Attempted to step a finished episode")
        transition = self.env.step(np.asarray(action, dtype=np.float32))
        self.last_transition = transition
        _, reward, terminated, truncated, info = transition
        self.elapsed_steps += 1
        self.episode_done = self._as_bool(terminated) or self._as_bool(truncated)
        if self.print_env_info:
            print(f"[{self.elapsed_steps:3}] reward={reward} info={info}")
        if self.vis:
            self.base_env.render_human()
        return transition

    def follow_path(self, result, refine_steps: int = 0):
        positions = self.validate_arm_path(result, stage=self.execution_stage)
        transition = self.last_transition
        for index in range(len(positions) + refine_steps):
            qpos = positions[min(index, len(positions) - 1)]
            transition = self._step(np.hstack([qpos, self.gripper_state]))
            if self.waypoint_validator is not None:
                self.waypoint_validator(
                    stage=self.execution_stage,
                    progress=(index + 1) / (len(positions) + refine_steps),
                )
            if self.episode_done:
                break
        return transition

    def _set_gripper(self, state: float, steps: int):
        if not -1.0 <= state <= 1.0:
            raise ValueError(f"Invalid normalized gripper command: {state}")
        if steps <= 0:
            raise ValueError(f"Gripper steps must be positive, got {steps}")
        self.gripper_state = state
        transition = self.last_transition
        for index in range(steps):
            qpos = self.robot.get_qpos()[0, :6].cpu().numpy()
            transition = self._step(np.hstack([qpos, self.gripper_state]))
            if self.waypoint_validator is not None:
                self.waypoint_validator(
                    stage=self.execution_stage,
                    progress=(index + 1) / steps,
                )
            if self.episode_done:
                break
        return transition

    def hold_current(self, steps: int):
        if steps <= 0:
            raise ValueError("hold steps must be positive")
        transition = self.last_transition
        for index in range(steps):
            qpos = self.robot.get_qpos()[0, :6].cpu().numpy()
            transition = self._step(np.hstack([qpos, self.gripper_state]))
            if self.waypoint_validator is not None:
                self.waypoint_validator(
                    stage=self.execution_stage,
                    progress=(index + 1) / steps,
                )
            if self.episode_done:
                break
        return transition

    def open_gripper(self, t: int = 6, gripper_state: float | None = None):
        return self._set_gripper(
            self.OPEN if gripper_state is None else gripper_state, t
        )

    def close_gripper(self, t: int = 6, gripper_state: float | None = None):
        return self._set_gripper(
            self.CLOSED if gripper_state is None else gripper_state, t
        )

    def move_to_joint_target(self, target_qpos: np.ndarray, steps: int = 20):
        target_qpos = np.asarray(target_qpos, dtype=np.float32)
        if target_qpos.shape != (6,) or not np.all(np.isfinite(target_qpos)):
            raise ValueError(f"Invalid PIPER arm target: {target_qpos}")
        if steps <= 0:
            raise ValueError(f"Joint interpolation steps must be positive, got {steps}")
        start_qpos = self.robot.get_qpos()[0, :6].cpu().numpy()
        transition = self.last_transition
        for target in np.linspace(start_qpos, target_qpos, steps + 1)[1:]:
            transition = self._step(np.hstack([target, self.gripper_state]))
            if self.episode_done:
                break
        return transition
