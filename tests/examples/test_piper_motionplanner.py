from types import SimpleNamespace
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import sapien
import torch

from mani_skill.examples.motionplanning.piper.motionplanner import (
    PiperMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.piper.solutions.lift_cube import (
    FULL_TILT_PLANAR_REACH,
    MAX_APPROACH_TILT,
    UNTILTED_PLANAR_REACH,
    build_adaptive_grasp_pose,
)
from mani_skill.examples.motionplanning.piper.solutions.lift_anything import (
    MAX_GRASP_CANDIDATES,
    generate_grasp_candidates,
)
from scripts.collect_lift_cube_piper import StrictEpisodeRecorder
from scripts.collect_lift_anything_piper import projected_bbox_size


class _ActionSpace:
    def contains(self, action):
        return action.shape == (7,) and np.all(action >= -2.0) and np.all(action <= 2.0)


class _FakeEnv(gym.Env):
    action_space = _ActionSpace()

    def __init__(self):
        super().__init__()
        self.agent = SimpleNamespace(
            robot=SimpleNamespace(get_qpos=lambda: np.zeros((1, 8), dtype=np.float32))
        )

    def reset(self, **kwargs):
        image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        return (
            {
                "sensor_data": {
                    uid: {"rgb": image}
                    for uid in ("base_camera", "wrist_camera", "side_camera")
                }
            },
            {},
        )

    def step(self, action):
        obs, _ = self.reset()
        return obs, 0.0, np.array([False]), np.array([False]), {}

    def close(self):
        pass


def test_recorder_records_observation_before_action(monkeypatch) -> None:
    fake = _FakeEnv()
    monkeypatch.setattr(
        "scripts.collect_lift_cube_piper._to_numpy",
        lambda value: (
            value if isinstance(value, np.ndarray) else value.detach().cpu().numpy()
        ),
    )
    recorder = StrictEpisodeRecorder(fake)
    recorder.reset(seed=1)
    action = np.zeros(7, dtype=np.float32)
    recorder.step(action)

    assert len(recorder.frames) == 1
    np.testing.assert_array_equal(recorder.frames[0]["action_command_raw"], action)
    np.testing.assert_array_equal(recorder.frames[0]["action_command_applied"], action)
    assert recorder.frames[0]["state"].shape == (7,)


def test_recorder_rejects_actions_that_would_be_clipped(monkeypatch) -> None:
    fake = _FakeEnv()
    monkeypatch.setattr(
        "scripts.collect_lift_cube_piper._to_numpy",
        lambda value: (
            value if isinstance(value, np.ndarray) else value.detach().cpu().numpy()
        ),
    )
    recorder = StrictEpisodeRecorder(fake)
    recorder.reset(seed=1)

    with pytest.raises(ValueError, match="would be clipped"):
        recorder.step(np.full(7, 3.0, dtype=np.float32))


def test_solver_boolean_conversion_is_scalar() -> None:
    assert PiperMotionPlanningSolver._as_bool(np.array([True]))
    assert not PiperMotionPlanningSolver._as_bool(np.array([False]))


def test_follow_path_rejects_joint_limit_violation_before_stepping() -> None:
    class NeverStepEnv:
        def step(self, action):
            raise AssertionError(f"invalid path reached env.step: {action}")

    qlimits = torch.tensor(
        [
            [
                [-2.618, 2.618],
                [0.0, 3.14],
                [-2.967, 0.0],
                [-1.745, 1.745],
                [-1.22, 1.22],
                [-2.0944, 2.0944],
                [0.0, 0.035],
                [-0.035, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    solver = object.__new__(PiperMotionPlanningSolver)
    solver.robot = SimpleNamespace(get_qlimits=lambda: qlimits)
    solver.env = NeverStepEnv()
    solver.execution_stage = "lift"
    solver.episode_done = False
    solver.last_transition = None
    solver.waypoint_validator = None
    solver.print_env_info = False
    solver.vis = False
    solver.gripper_state = -1.0
    result = {
        "position": np.array(
            [
                [
                    -0.34731367,
                    1.5157384,
                    -1.2022141,
                    -0.04970372,
                    1.2235266,
                    0.9708683,
                ]
            ]
        )
    }

    with pytest.raises(RuntimeError, match="lift: joint-limit-joint5"):
        solver.follow_path(result)


def test_planning_urdf_preserves_collision_and_removes_render_geometry(
    tmp_path,
) -> None:
    source = tmp_path / "robot.urdf"
    source.write_text("""<?xml version="1.0"?>
<robot name="test">
  <link name="base">
    <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
    <visual><geometry><box size="1 1 1"/></geometry></visual>
    <collision><geometry><box size="1 1 1"/></geometry></collision>
  </link>
</robot>
""")
    source.with_suffix(".srdf").write_text('<robot name="test"/>')

    generated = PiperMotionPlanningSolver._build_kinematic_planning_urdf(str(source))
    root = ET.parse(generated).getroot()

    assert root.find(".//inertial") is None
    assert root.find(".//visual") is None
    assert root.find(".//collision") is not None
    assert root.find("./link").attrib["name"] == "base"


def test_planning_targets_are_converted_to_robot_base_frame() -> None:
    solver = object.__new__(PiperMotionPlanningSolver)
    solver.base_pose = sapien.Pose([-0.35, 0.0, 0.0])
    target = sapien.Pose([0.03, 0.02, 0.08])

    transformed = solver._transform_pose_for_planning(target)

    np.testing.assert_allclose(transformed.p, [0.38, 0.02, 0.08], atol=1e-6)


def test_collision_point_cloud_is_converted_to_robot_base_frame() -> None:
    solver = object.__new__(PiperMotionPlanningSolver)
    solver.base_pose = sapien.Pose([-0.35, 0.0, 0.0])
    recorded = {}
    solver.planner = SimpleNamespace(
        update_point_cloud=lambda points, radius: recorded.update(
            points=points, radius=radius
        )
    )

    solver.set_collision_point_cloud(np.array([[0.0, 0.0, 0.0]]), radius=0.004)

    np.testing.assert_allclose(recorded["points"], [[0.35, 0.0, 0.0]])
    assert recorded["radius"] == 0.004


def test_piper_collision_proxy_supports_self_collision_queries() -> None:
    import mplib

    source = Path("mani_skill/assets/robots/piper/piper_description.urdf")
    generated = PiperMotionPlanningSolver._build_collision_planning_urdf(str(source))
    generated_root = ET.parse(generated).getroot()
    collision_counts = {
        link.attrib["name"]: len(link.findall("collision"))
        for link in generated_root.findall("link")
    }
    assert collision_counts["gripper_base"] > 1
    assert collision_counts["link7"] > 1
    assert collision_counts["link8"] > 1
    assert not any(
        ".repaired." in mesh.attrib["filename"]
        for mesh in generated_root.findall(".//collision/geometry/mesh")
    )
    links = [
        "base_link",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "gripper_base",
        "link7",
        "link8",
        "piper_tcp",
    ]
    joints = [f"joint{index}" for index in range(1, 9)]
    planner = mplib.Planner(
        urdf=str(generated),
        srdf=str(source.with_suffix(".srdf")),
        user_link_names=links,
        user_joint_names=joints,
        move_group="piper_tcp",
    )

    safe = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0, 0.035, -0.035])
    colliding = np.array(
        [-2.5304, 1.5498, -0.0843, -0.7861, 0.6057, -0.3593, 0.0073, -0.0033]
    )

    assert planner.check_for_self_collision(qpos=safe) == []
    collision_pairs = {
        (item.link_name1, item.link_name2)
        for item in planner.check_for_self_collision(qpos=colliding)
    }
    assert ("base_link", "gripper_base") in collision_pairs


def test_joint_target_validation_rejects_wrong_shape() -> None:
    solver = object.__new__(PiperMotionPlanningSolver)
    with pytest.raises(ValueError, match="Invalid PIPER arm target"):
        solver.move_to_joint_target(np.zeros(7, dtype=np.float32))


class _GraspPoseAgent:
    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        return approaching, closing, center


@pytest.mark.parametrize(
    ("planar_reach", "expected_tilt"),
    [
        (UNTILTED_PLANAR_REACH, 0.0),
        (FULL_TILT_PLANAR_REACH, MAX_APPROACH_TILT),
    ],
)
def test_adaptive_grasp_pose_is_orthonormal(planar_reach, expected_tilt) -> None:
    approaching, closing, center = build_adaptive_grasp_pose(
        _GraspPoseAgent(),
        np.array([planar_reach, 0.0, 0.02]),
        np.zeros(3),
    )

    np.testing.assert_allclose(np.linalg.norm(approaching), 1.0, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(closing), 1.0, atol=1e-7)
    np.testing.assert_allclose(approaching @ closing, 0.0, atol=1e-7)
    np.testing.assert_allclose(np.arccos(-approaching[2]), expected_tilt, atol=1e-7)
    np.testing.assert_array_equal(center, [planar_reach, 0.0, 0.02])


class _BoundingBox:
    def __init__(self, extents):
        self.bounds = np.array([np.zeros(3), extents], dtype=np.float64)


class _Mesh:
    def __init__(self, extents):
        self.bounding_box = _BoundingBox(extents)


class _Object:
    def __init__(self, extents):
        self._mesh = _Mesh(extents)
        self.requested_world_frame = None
        self.pose = SimpleNamespace(
            to_transformation_matrix=lambda: torch.eye(4)[None, :]
        )

    def get_first_collision_mesh(self, *, to_world_frame=True):
        self.requested_world_frame = to_world_frame
        return self._mesh


class _AnythingAgent:
    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        np.testing.assert_allclose(np.linalg.norm(approaching), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(closing), 1.0, atol=1e-6)
        np.testing.assert_allclose(approaching @ closing, 0.0, atol=1e-6)
        return sapien.Pose(center)


def test_liftanything_candidates_are_bounded_and_width_feasible() -> None:
    obj = _Object([0.04, 0.06, 0.08])
    candidates = generate_grasp_candidates(
        _AnythingAgent(),
        obj,
        np.array([-0.35, 0.0, 0.0]),
    )

    assert obj.requested_world_frame is False
    assert len(candidates) == MAX_GRASP_CANDIDATES
    assert all(candidate.required_width <= 0.068 for candidate in candidates)
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)


def test_liftanything_candidates_reject_width_infeasible_object() -> None:
    with pytest.raises(RuntimeError, match="width-infeasible"):
        generate_grasp_candidates(
            _AnythingAgent(),
            _Object([0.08, 0.09, 0.04]),
            np.array([-0.35, 0.0, 0.0]),
        )


def test_liftanything_candidates_reject_width_above_executable_contract() -> None:
    with pytest.raises(RuntimeError, match="width-infeasible"):
        generate_grasp_candidates(
            _AnythingAgent(),
            _Object([0.069, 0.09, 0.04]),
            np.array([-0.35, 0.0, 0.0]),
        )


def test_projected_bbox_size_uses_opencv_extrinsics() -> None:
    points = np.array(
        [
            [-0.1, -0.1, 1.0],
            [0.1, -0.1, 1.0],
            [-0.1, 0.1, 1.0],
            [0.1, 0.1, 1.0],
        ]
    )
    intrinsic = np.array([[100.0, 0.0, 112.0], [0.0, 100.0, 112.0], [0.0, 0.0, 1.0]])
    extrinsic = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)

    width, height = projected_bbox_size(points, extrinsic, intrinsic)

    assert width == pytest.approx(20.0)
    assert height == pytest.approx(20.0)
