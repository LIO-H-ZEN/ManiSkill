import argparse
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from mani_skill.envs.tasks.pick_anything.episode_specs import EpisodeSpec, ObjectSpec
from scripts import run_piper_lift_tasks as launcher


def _args(**overrides) -> argparse.Namespace:
    values = {
        "task": "both",
        "mode": "reset",
        "seed": 7,
        "num_envs": 1,
        "render_backend": "cuda:0",
        "object_source": None,
        "table_randomizer": "wood",
        "floor_randomizer": "grid",
        "clutter": None,
        "clutter_sources": None,
        "domain_rand_freq": 25,
        "domain_rand_axes": None,
        "episode_spec": None,
        "episode_index": 0,
        "vis": False,
        "video_dir": pathlib.Path("/tmp/piper-lift-test-videos"),
        "video_fps": 20,
        "video_view": "triptych",
        "task_prompt": None,
        "no_video": True,
        "video_run_dir": pathlib.Path("/tmp/piper-lift-test-videos/test-run"),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _episode_spec() -> EpisodeSpec:
    return EpisodeSpec(
        stable_episode_id="cube-launch-000001",
        environment_seed=17,
        object_spec=ObjectSpec(
            source="cube",
            object_id="cube-launch",
            cube_half_size=0.02,
            cube_color=(1.0, 0.0, 0.0, 1.0),
        ),
        object_position=(0.03, 0.0, 0.02),
        object_quaternion=(1.0, 0.0, 0.0, 0.0),
        table_kind="WoodTableRandomizer",
        table_texture=None,
        table_friction=None,
        floor_texture=None,
        hdri=None,
        directional_light_direction=(1.0, 0.0, -1.0),
        directional_light_intensity=0.8,
        robot_init_qpos=(0.0, 1.57, -1.3485, 0.0, 0.0, 0.0, 0.035, -0.035),
    )


def _observation() -> dict:
    image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    return {
        "sensor_data": {
            camera_uid: {"rgb": image.copy()} for camera_uid in launcher.CAMERA_UIDS
        }
    }


def test_environment_kwargs_lock_the_runtime_contract() -> None:
    env_id, kwargs = launcher._environment_kwargs("liftcube", _args())

    assert env_id == "LiftCubePiper-v1"
    assert kwargs == {
        "robot_uids": "piper_wristcam",
        "obs_mode": "rgb",
        "control_mode": "pd_joint_pos",
        "sim_backend": "physx_cpu",
        "num_envs": 1,
        "max_episode_steps": 100,
        "render_backend": "cuda:0",
    }

    env_id, kwargs = launcher._environment_kwargs("liftanything", _args())
    assert env_id == "LiftAnythingPiper-v1"
    assert kwargs["object_sources"] == ("cube",)
    assert kwargs["table_randomizer"] == "wood"
    assert kwargs["floor_randomizer"] == "grid"
    assert kwargs["domain_rand_freq"] == 25
    assert "domain_rand_axes" not in kwargs

    _, kwargs = launcher._environment_kwargs(
        "liftcube", _args(mode="expert", no_video=False)
    )
    assert kwargs["render_mode"] == "sensors"

    _, kwargs = launcher._environment_kwargs(
        "liftcube",
        _args(mode="expert", no_video=False, video_view="third-person"),
    )
    assert kwargs["render_mode"] == "rgb_array"


def test_video_recorder_class_matches_requested_view() -> None:
    assert (
        launcher._video_recorder_class("triptych") is launcher.ThreeCameraRecordEpisode
    )
    assert (
        launcher._video_recorder_class("third-person") is launcher.PromptedRecordEpisode
    )
    with pytest.raises(ValueError, match="Unsupported video view"):
        launcher._video_recorder_class("unknown")


def test_camera_triptych_has_labeled_panels_in_locked_order() -> None:
    sensor_images = {
        "base_camera": {
            "rgb": np.full((1, 224, 224, 3), 10, dtype=np.uint8),
        },
        "wrist_camera": {
            "rgb": np.full((1, 224, 224, 3), 20, dtype=np.uint8),
        },
        "side_camera": {
            "rgb": np.full((1, 224, 224, 3), 30, dtype=np.uint8),
        },
    }

    triptych = launcher._camera_triptych(sensor_images)

    assert triptych.shape == (224, 672, 3)
    assert triptych.dtype == np.uint8
    assert np.all(triptych[200, 100] == 10)
    assert np.all(triptych[200, 324] == 20)
    assert np.all(triptych[200, 548] == 30)
    assert np.max(triptych[:40, :224]) > 200
    assert np.max(triptych[:40, 224:448]) > 200
    assert np.max(triptych[:40, 448:]) > 200


def test_task_prompt_is_overlaid_without_resizing_the_frame() -> None:
    image = np.full((224, 672, 3), 17, dtype=np.uint8)

    prompted = launcher._with_task_prompt(image, "Pick up the target object.")

    assert prompted.shape == image.shape
    assert prompted.dtype == np.uint8
    assert np.all(prompted[:170] == 17)
    assert np.min(prompted[170:]) < 17
    assert np.max(prompted[170:]) > 200


def test_long_task_prompt_wraps_inside_the_original_frame() -> None:
    image = np.full((224, 224, 3), 17, dtype=np.uint8)
    short = launcher._with_task_prompt(image, "Pick up the mug.")
    long = launcher._with_task_prompt(
        image,
        "Pick up the chicken leg without touching either distractor object.",
    )

    short_changed_rows = np.flatnonzero(np.any(short != image, axis=(1, 2)))
    long_changed_rows = np.flatnonzero(np.any(long != image, axis=(1, 2)))
    assert long.shape == image.shape
    assert np.ptp(long_changed_rows) > np.ptp(short_changed_rows)
    assert np.all(long[:120] == 17)


def test_empty_task_prompt_fast_fails() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        launcher._with_task_prompt(np.zeros((224, 672, 3), dtype=np.uint8), "   ")


@pytest.mark.parametrize(
    ("actor_name", "expected_prompt"),
    [
        ("cube-0", "Pick up the cube."),
        ("ycb-003_cracker_box-0", "Pick up the cracker box."),
        (
            "interndata-omniobject3d-chicken_leg_005-0",
            "Pick up the chicken leg.",
        ),
        ("interndata-google_scan-toy_bus_001-0", "Pick up the toy bus."),
    ],
)
def test_default_liftanything_prompt_names_sampled_object(
    actor_name, expected_prompt
) -> None:
    env = SimpleNamespace(unwrapped=SimpleNamespace(object_names=[actor_name]))

    assert launcher._default_task_prompt("liftanything", env) == expected_prompt


def test_default_liftanything_prompt_fast_fails_for_unknown_actor_name() -> None:
    env = SimpleNamespace(unwrapped=SimpleNamespace(object_names=["mystery-0"]))

    with pytest.raises(RuntimeError, match="pass --task-prompt"):
        launcher._default_task_prompt("liftanything", env)


def test_episode_spec_file_is_selected_by_index(tmp_path) -> None:
    first = _episode_spec()
    second = EpisodeSpec.from_dict(
        {
            **first.to_dict(),
            "stable_episode_id": "cube-launch-000002",
            "environment_seed": 18,
        }
    )
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps({"episodes": [first.to_dict(), second.to_dict()]}))

    env_id, kwargs = launcher._environment_kwargs(
        "liftanything",
        _args(episode_spec=path, episode_index=1),
    )

    assert env_id == "LiftAnythingPiper-v1"
    assert kwargs["episode_spec"]["stable_episode_id"] == "cube-launch-000002"
    assert "object_sources" not in kwargs
    assert "table_randomizer" not in kwargs
    assert "floor_randomizer" not in kwargs


def test_conflicting_liftanything_inputs_fast_fail(tmp_path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        launcher._validate_args(
            _args(episode_spec=tmp_path / "episodes.json", object_source="cube")
        )

    with pytest.raises(ValueError, match="requires enabled --clutter"):
        launcher._validate_args(_args(clutter_sources=["interndata"]))

    with pytest.raises(ValueError, match="only valid for LiftAnything"):
        launcher._validate_args(
            _args(task="liftcube", clutter="2", clutter_sources=["interndata"])
        )


def test_parallel_launcher_is_reset_only_and_rejects_episode_specs(tmp_path) -> None:
    launcher._validate_args(_args(num_envs=8, mode="reset"))

    with pytest.raises(ValueError, match="only supported with --mode reset"):
        launcher._validate_args(_args(num_envs=8, mode="expert"))
    with pytest.raises(ValueError, match="requires --num-envs 1"):
        launcher._validate_args(
            _args(num_envs=8, episode_spec=tmp_path / "episodes.json")
        )


def test_liftanything_randomization_options_are_forwarded() -> None:
    _, kwargs = launcher._environment_kwargs(
        "liftanything",
        _args(
            num_envs=8,
            object_source="cube",
            table_randomizer="procedural",
            floor_randomizer="grid",
            clutter="random_2_5",
            clutter_sources=["interndata"],
            domain_rand_freq=10,
            domain_rand_axes=["table", "clutter"],
        ),
    )

    assert kwargs["num_envs"] == 8
    assert kwargs["sim_backend"] == "physx_cuda"
    assert kwargs["clutter"] == "random_2_5"
    assert kwargs["clutter_sources"] == ("interndata",)
    assert kwargs["domain_rand_freq"] == 10
    assert kwargs["domain_rand_axes"] == ("table", "clutter")
    assert kwargs["table_randomizer"] == "procedural"


def test_camera_contract_requires_all_three_uint8_images() -> None:
    shapes = launcher._camera_contract(_observation())
    assert shapes == {
        camera_uid: [1, 224, 224, 3] for camera_uid in launcher.CAMERA_UIDS
    }

    invalid = _observation()
    invalid["sensor_data"]["side_camera"]["rgb"] = np.zeros(
        (1, 128, 128, 3), dtype=np.uint8
    )
    with pytest.raises(RuntimeError, match="side_camera violates"):
        launcher._camera_contract(invalid)


def test_camera_contract_accepts_parallel_batches() -> None:
    observation = _observation()
    for camera_uid in launcher.CAMERA_UIDS:
        observation["sensor_data"][camera_uid]["rgb"] = np.zeros(
            (4, 224, 224, 3), dtype=np.uint8
        )

    shapes = launcher._camera_contract(observation, num_envs=4)

    assert shapes == {
        camera_uid: [4, 224, 224, 3] for camera_uid in launcher.CAMERA_UIDS
    }


def test_run_task_resets_with_seed_and_always_closes(monkeypatch) -> None:
    class FakeEnv:
        def __init__(self):
            self.reset_calls = []
            self.closed = False

        def reset(self, **kwargs):
            self.reset_calls.append(kwargs)
            return _observation(), {}

        def close(self):
            self.closed = True

    env = FakeEnv()
    monkeypatch.setattr(launcher.gym, "make", lambda *args, **kwargs: env)

    summary = launcher.run_task("liftcube", _args())

    assert env.reset_calls == [{"seed": 7, "options": {"reconfigure": True}}]
    assert env.closed
    assert summary["environment"] == "LiftCubePiper-v1"
    assert summary["mode"] == "reset"


def test_run_task_closes_after_expert_failure(monkeypatch) -> None:
    class FakeEnv:
        closed = False

        def reset(self, **kwargs):
            return _observation(), {}

        def close(self):
            self.closed = True

    env = FakeEnv()
    monkeypatch.setattr(launcher.gym, "make", lambda *args, **kwargs: env)
    monkeypatch.setattr(
        launcher,
        "_run_expert",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("planner failed")),
    )

    with pytest.raises(RuntimeError, match="planner failed"):
        launcher.run_task("liftcube", _args(mode="expert"))
    assert env.closed


def test_run_task_records_expert_video(monkeypatch, tmp_path) -> None:
    class FakeEnv:
        closed = False

        def reset(self, **kwargs):
            return _observation(), {}

        def close(self):
            self.closed = True

    recorded = {}

    class FakeRecorder:
        def __init__(self, env, **kwargs):
            self.env = env
            self.output_dir = pathlib.Path(kwargs["output_dir"])
            recorded.update(kwargs)

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

        def close(self):
            self.output_dir.mkdir(parents=True)
            (self.output_dir / "0.mp4").write_bytes(b"video")
            self.env.close()

    env = FakeEnv()
    monkeypatch.setattr(launcher.gym, "make", lambda *args, **kwargs: env)
    monkeypatch.setattr(launcher, "ThreeCameraRecordEpisode", FakeRecorder)
    monkeypatch.setattr(
        launcher,
        "_run_expert",
        lambda *args, **kwargs: {"success": True, "reason": "accepted"},
    )

    summary = launcher.run_task(
        "liftcube",
        _args(
            mode="expert",
            no_video=False,
            video_run_dir=tmp_path / "run",
        ),
    )

    assert recorded["save_video"] is True
    assert recorded["save_trajectory"] is False
    assert recorded["video_fps"] == 20
    assert recorded["task_prompt"] == "Pick up the red cube."
    assert summary["video_view"] == "triptych"
    assert summary["task_prompt"] == "Pick up the red cube."
    assert summary["video_files"] == [
        str(tmp_path / "run" / "LiftCubePiper-v1" / "0.mp4")
    ]
    assert env.closed


def test_custom_task_prompt_is_reported(monkeypatch) -> None:
    class FakeEnv:
        def reset(self, **kwargs):
            return _observation(), {}

        def close(self):
            pass

    monkeypatch.setattr(launcher.gym, "make", lambda *args, **kwargs: FakeEnv())

    summary = launcher.run_task(
        "liftanything", _args(task="liftanything", task_prompt="Lift the mug.")
    )

    assert summary["task_prompt"] == "Lift the mug."
