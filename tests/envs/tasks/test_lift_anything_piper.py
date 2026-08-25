import dataclasses
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mani_skill.envs.tasks.pick_anything.pick_anything_env as pick_anything_module
from mani_skill.envs.tasks.pick_anything.episode_specs import (
    EpisodeSpec,
    ObjectSpec,
    SettledObjectState,
    load_episode_specs_manifest,
)
from mani_skill.envs.tasks.pick_anything.lift_anything_piper import (
    MAX_PLANAR_REACH,
    LiftAnythingPiperEnv,
    validate_settled_spawn,
)
from mani_skill.envs.tasks.pick_anything.pick_anything_env import PickAnythingEnv
from mani_skill.envs.tasks.pick_anything.randomization.clutter_randomizer import (
    ClutterRandomizer,
)
from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG
from mani_skill.envs.tasks.pick_anything.randomization.object_randomizer import (
    CompositeObjectRandomizer,
)
from mani_skill.envs.tasks.tabletop.lift_cube_piper import LIFT_HEIGHT
from scripts.build_lift_anything_supplement import build_supplemental_specs
from scripts.materialize_lift_anything_episode_specs import (
    _materialize,
    materialization_coordinates,
    select_object_specs,
)


def _episode_spec() -> EpisodeSpec:
    return EpisodeSpec(
        stable_episode_id="cube-formal-0001",
        environment_seed=17,
        object_spec=ObjectSpec(
            source="cube",
            object_id="cube-0001",
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


def test_episode_spec_round_trip_and_fingerprint() -> None:
    spec = _episode_spec()
    restored = EpisodeSpec.from_dict(spec.to_dict())

    assert restored == spec
    assert restored.fingerprint == spec.fingerprint
    assert len(spec.fingerprint) == 64


def test_episode_spec_round_trips_optional_settled_state() -> None:
    state = SettledObjectState(
        position=(0.01, 0.02, 0.03),
        quaternion=(-1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.001, 0.0, 0.0),
        angular_velocity=(0.0, 0.002, 0.0),
    )
    spec = dataclasses.replace(_episode_spec(), settled_object_state=state)

    restored = EpisodeSpec.from_dict(spec.to_dict())

    assert restored == spec
    assert restored.settled_object_state.quaternion == (1.0, -0.0, -0.0, -0.0)
    assert len(restored.settled_object_state.fingerprint) == 64


def test_settled_state_replay_mismatch_fast_fails() -> None:
    expected = SettledObjectState(
        position=(0.01, 0.02, 0.03),
        quaternion=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
    )
    actual = dataclasses.replace(expected, position=(0.011, 0.02, 0.03))

    with pytest.raises(RuntimeError, match="settled position"):
        LiftAnythingPiperEnv._assert_settled_object_state(expected, actual)


def test_benchmark_manifest_requires_settled_state(tmp_path) -> None:
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps({"episodes": [_episode_spec().to_dict()]}))

    with pytest.raises(ValueError, match="post-settle rigid-body state"):
        load_episode_specs_manifest(path, require_settled_state=True)


def test_legacy_manifest_remains_readable_without_settled_state(tmp_path) -> None:
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps({"episodes": [_episode_spec().to_dict()]}))

    assert load_episode_specs_manifest(path) == [_episode_spec()]


def test_object_spec_fast_fails_invalid_source_fields() -> None:
    with pytest.raises(ValueError, match="requires category"):
        ObjectSpec(source="interndata", object_id="instance")
    with pytest.raises(ValueError, match="cube fields"):
        ObjectSpec(source="ycb", object_id="003_cracker_box", cube_half_size=0.02)


def test_composite_randomizer_can_disable_goal() -> None:
    randomizer = CompositeObjectRandomizer(["cube"], create_goal=False)
    assert not randomizer.create_goal


def test_composite_randomizer_exposes_concrete_object_names() -> None:
    randomizer = CompositeObjectRandomizer(["cube"], create_goal=False)

    class FakeSource:
        name = "interndata"

        def build_actor(self, env, env_idx, rng):
            return SimpleNamespace(name="interndata-omniobject3d-chicken_leg_005-0")

    randomizer.sources = [FakeSource()]
    env = SimpleNamespace(
        num_envs=1,
        _batched_episode_rng=BatchedRNG.from_seeds(np.array([0])),
        remove_from_state_dict_registry=lambda actor: None,
        add_to_state_dict_registry=lambda actor: None,
    )

    randomizer.on_reconfigure(env, {})

    assert env.object_sources == ["interndata"]
    assert env.object_names == ["interndata-omniobject3d-chicken_leg_005-0"]


def test_spawn_validation_rejects_excess_motion_and_reach() -> None:
    validate_settled_spawn(
        initial_position=np.array([0.0, 0.0, 0.02]),
        settled_position=np.array([0.14, 0.0, 0.02]),
        settled_bottom_z=0.0,
        robot_base_position=np.array([-0.2, 0.0, 0.0]),
    )
    with pytest.raises(RuntimeError, match="moved"):
        validate_settled_spawn(
            initial_position=np.array([0.0, 0.0, 0.02]),
            settled_position=np.array([0.16, 0.0, 0.02]),
            settled_bottom_z=0.0,
            robot_base_position=np.array([-0.35, 0.0, 0.0]),
        )
    with pytest.raises(RuntimeError, match="planar reach"):
        validate_settled_spawn(
            initial_position=np.array([MAX_PLANAR_REACH + 0.01, 0.0, 0.02]),
            settled_position=np.array([MAX_PLANAR_REACH + 0.01, 0.0, 0.02]),
            settled_bottom_z=0.0,
            robot_base_position=np.zeros(3),
        )


def test_spawn_validation_uses_settled_world_bottom() -> None:
    validate_settled_spawn(
        initial_position=np.array([0.0, 0.0, 0.07]),
        settled_position=np.array([0.0, 0.0, 0.02]),
        settled_bottom_z=-0.001,
        robot_base_position=np.array([-0.35, 0.0, 0.0]),
    )
    with pytest.raises(RuntimeError, match="penetrated"):
        validate_settled_spawn(
            initial_position=np.array([0.0, 0.0, 0.07]),
            settled_position=np.array([0.0, 0.0, 0.02]),
            settled_bottom_z=-0.004,
            robot_base_position=np.array([-0.35, 0.0, 0.0]),
        )


def test_evaluate_uses_relative_settled_height_and_grasp() -> None:
    env = object.__new__(LiftAnythingPiperEnv)
    env.object_rest_z = torch.tensor([0.04, 0.03])
    env.obj = SimpleNamespace(
        pose=SimpleNamespace(
            p=torch.tensor(
                [
                    [0.0, 0.0, 0.04 + LIFT_HEIGHT],
                    [0.0, 0.0, 0.03 + LIFT_HEIGHT],
                ]
            )
        )
    )
    env.agent = SimpleNamespace(is_grasping=lambda obj: torch.tensor([True, False]))
    env.success_streak = torch.tensor([2, 2], dtype=torch.int32)
    env.streak_updated_at = torch.tensor([2, 2], dtype=torch.int32)
    env._elapsed_steps = torch.tensor([3, 3], dtype=torch.int32)
    env.success_streak_steps = 3

    info = env.evaluate()

    torch.testing.assert_close(info["success"], torch.tensor([True, False]))
    torch.testing.assert_close(
        info["legacy_success_3step"], torch.tensor([True, False])
    )
    torch.testing.assert_close(
        info["robust_success_10step"], torch.tensor([False, False])
    )
    torch.testing.assert_close(
        info["success_streak"], torch.tensor([3, 0], dtype=torch.int32)
    )


def test_evaluate_can_require_robust_ten_step_hold() -> None:
    env = object.__new__(LiftAnythingPiperEnv)
    env.object_rest_z = torch.tensor([0.04])
    env.obj = SimpleNamespace(
        pose=SimpleNamespace(p=torch.tensor([[0.0, 0.0, 0.04 + LIFT_HEIGHT]]))
    )
    env.agent = SimpleNamespace(is_grasping=lambda obj: torch.tensor([True]))
    env.success_streak = torch.tensor([8], dtype=torch.int32)
    env.streak_updated_at = torch.tensor([8], dtype=torch.int32)
    env._elapsed_steps = torch.tensor([9], dtype=torch.int32)
    env.success_streak_steps = 10

    ninth = env.evaluate()
    env._elapsed_steps = torch.tensor([10], dtype=torch.int32)
    tenth = env.evaluate()

    assert not bool(ninth["success"][0])
    assert bool(ninth["legacy_success_3step"][0])
    assert bool(tenth["success"][0])
    assert bool(tenth["robust_success_10step"][0])


def test_environment_is_registered_with_locked_horizon() -> None:
    from mani_skill.utils.registration import REGISTERED_ENVS

    assert REGISTERED_ENVS["LiftAnythingPiper-v1"].max_episode_steps == 100


def test_randomization_defaults_match_pickanything() -> None:
    parameters = inspect.signature(LiftAnythingPiperEnv.__init__).parameters

    assert parameters["num_envs"].default == 1
    assert parameters["domain_rand_freq"].default == 25
    assert parameters["domain_rand_axes"].default is None
    assert parameters["clutter"].default is None
    assert parameters["clutter_sources"].default is None


def test_pickanything_builds_clutter_from_independent_source_pool(monkeypatch) -> None:
    captured = {}

    class FakeObjectRandomizer:
        def __init__(self, sources, **kwargs):
            captured["target_sources"] = list(sources)

    class FakeClutterRandomizer:
        def __init__(self, *, num_clutter, sources):
            captured["clutter_count"] = num_clutter
            captured["clutter_sources"] = list(sources)

    monkeypatch.setattr(
        pick_anything_module, "CompositeObjectRandomizer", FakeObjectRandomizer
    )
    monkeypatch.setattr(
        pick_anything_module, "ClutterRandomizer", FakeClutterRandomizer
    )
    monkeypatch.setattr(
        pick_anything_module, "resolve_table_randomizer", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        pick_anything_module, "resolve_floor_randomizer", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        pick_anything_module.BaseEnv, "__init__", lambda *args, **kwargs: None
    )

    PickAnythingEnv(
        object_sources=("cube",),
        clutter=2,
        clutter_sources=("interndata",),
        lighting_randomizer=object(),
    )

    assert captured == {
        "target_sources": ["cube"],
        "clutter_count": (2, 2),
        "clutter_sources": ["interndata"],
    }


def test_constructor_forwards_parallel_randomization_options(monkeypatch) -> None:
    captured = {}

    def fake_init(self, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(PickAnythingEnv, "__init__", fake_init)

    LiftAnythingPiperEnv(
        num_envs=8,
        object_sources=("cube",),
        table_randomizer="procedural",
        floor_randomizer="grid",
        clutter="random_2_5",
        clutter_sources=("interndata",),
        domain_rand_freq=10,
        domain_rand_axes=("table", "clutter"),
    )

    assert captured["num_envs"] == 8
    assert captured["object_sources"] == ("cube",)
    assert captured["table_randomizer"] == "procedural"
    assert captured["floor_randomizer"] == "grid"
    assert captured["clutter"] == "random_2_5"
    assert captured["clutter_sources"] == ("interndata",)
    assert captured["domain_rand_freq"] == 10
    assert captured["domain_rand_axes"] == ("table", "clutter")


def test_episode_spec_fast_fails_in_parallel_mode() -> None:
    with pytest.raises(ValueError, match="episode_spec requires num_envs=1"):
        LiftAnythingPiperEnv(episode_spec=_episode_spec(), num_envs=2)


def test_object_spec_can_be_replicated_in_parallel(monkeypatch) -> None:
    captured = {}

    def fake_init(self, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(PickAnythingEnv, "__init__", fake_init)
    spec = _episode_spec().object_spec

    LiftAnythingPiperEnv(object_spec=spec, num_envs=4, clutter=2)

    assert captured["num_envs"] == 4
    assert captured["clutter"] == 2
    assert captured["object_sources"] is None
    assert len(captured["clutter_sources"]) == 1
    assert captured["clutter_sources"][0].spec == spec


def test_object_spec_allows_independent_clutter_sources(monkeypatch) -> None:
    captured = {}

    def fake_init(self, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(PickAnythingEnv, "__init__", fake_init)

    LiftAnythingPiperEnv(
        object_spec=_episode_spec().object_spec,
        clutter=2,
        clutter_sources=("interndata",),
    )

    assert captured["clutter_sources"] == ("interndata",)


def test_parallel_episode_initialization_runs_all_randomizer_hooks() -> None:
    events = []

    class Hook:
        def __init__(self, name):
            self.name = name

        def on_initialize_episode(self, env, env_idx, options):
            events.append((self.name, env_idx.tolist()))

    class Agent:
        def __init__(self):
            self.keyframes = {
                "home": SimpleNamespace(
                    qpos=np.array([0.0, 1.57, -1.35, 0.0, 0.0, 0.0, 0.035, -0.035])
                )
            }
            self.robot = SimpleNamespace(set_pose=lambda pose: None)
            self.reset_qpos = None

        def reset(self, qpos):
            self.reset_qpos = np.asarray(qpos)

    env = object.__new__(LiftAnythingPiperEnv)
    env.num_envs = 3
    env.device = torch.device("cpu")
    env.episode_spec = None
    env.robot_init_qpos_noise = 0.02
    env.table_randomizer = Hook("table")
    env.floor_randomizer = Hook("floor")
    env.object_randomizer = Hook("object")
    env.clutter_randomizer = Hook("clutter")
    env.lighting_randomizer = Hook("lighting")
    env.agent = Agent()
    env.obj = SimpleNamespace(
        pose=SimpleNamespace(
            p=torch.tensor([[0.0, 0.0, 0.02], [0.0, 0.0, 0.03], [0.0, 0.0, 0.04]]),
            q=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3),
        )
    )
    env._batched_episode_rng = BatchedRNG.from_seeds(np.array([1, 2, 3]))
    env.object_rest_z = torch.full((3,), -1.0)
    env.success_streak = torch.full((3,), 9, dtype=torch.int32)
    env.streak_updated_at = torch.full((3,), 9, dtype=torch.int32)
    env._elapsed_steps = torch.full((3,), 9, dtype=torch.int32)

    env._initialize_episode(torch.tensor([0, 2]), {})

    assert events == [
        ("table", [0, 2]),
        ("floor", [0, 2]),
        ("object", [0, 2]),
        ("clutter", [0, 2]),
        ("lighting", [0, 2]),
    ]
    assert env.agent.reset_qpos.shape == (2, 8)
    torch.testing.assert_close(env.object_rest_z, torch.tensor([0.02, -1.0, 0.04]))
    torch.testing.assert_close(
        env.success_streak, torch.tensor([0, 9, 0], dtype=torch.int32)
    )
    torch.testing.assert_close(
        env._elapsed_steps, torch.tensor([0, 9, 0], dtype=torch.int32)
    )


def test_mid_episode_randomization_uses_per_env_cadence_and_selected_axes() -> None:
    calls = []

    class Hook:
        def __init__(self, name):
            self.name = name

        def on_step(self, env, env_idx, options):
            calls.append((self.name, env_idx.tolist()))

    env = object.__new__(LiftAnythingPiperEnv)
    env.domain_rand_freq = 25
    env.domain_rand_axes = {"table", "clutter"}
    env._elapsed_steps = torch.tensor([24, 23, 49])
    env.num_envs = 3
    env.device = torch.device("cpu")
    env.lighting_randomizer = Hook("lighting")
    env.table_randomizer = Hook("table")
    env.clutter_randomizer = Hook("clutter")
    env.scene = SimpleNamespace(gpu_sim_enabled=False)

    env._after_control_step()

    assert calls == [("table", [0, 2]), ("clutter", [0, 2])]


def test_cpu_clutter_partial_relocation_only_updates_selected_environment() -> None:
    class Actor:
        def __init__(self):
            self.pose = None

        def set_pose(self, pose):
            self.pose = pose

    randomizer = object.__new__(ClutterRandomizer)
    randomizer.lo = 2
    randomizer.hi = 2
    randomizer.spawn_half_size = 0.1
    randomizer.spawn_center = (0.0, 0.0)
    randomizer.min_dist = 0.06
    randomizer.max_resamples = 5
    actors = [Actor() for _ in range(6)]
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=3,
        gpu_sim_enabled=False,
        obj=SimpleNamespace(
            pose=SimpleNamespace(p=torch.tensor([[0.0, 0.0, 0.02]] * 3))
        ),
        clutter_zs=torch.full((6,), 0.02),
        _clutter_objs=actors,
    )

    randomizer._relocate_clutter(env, torch.tensor([1]))

    assert actors[0].pose is None
    assert actors[1].pose is None
    assert actors[2].pose is not None
    assert actors[3].pose is not None
    assert actors[4].pose is None
    assert actors[5].pose is None


def test_episode_materialization_retries_only_spawn_failures(monkeypatch) -> None:
    class FakeEnv:
        def __init__(self):
            self.unwrapped = self
            self.seeds = []
            self.closed = False

        def reset(self, *, seed, options):
            self.seeds.append(seed)
            if len(self.seeds) == 1:
                raise RuntimeError("spawn-invalid: unstable pose")

        def capture_episode_spec(self, stable_episode_id):
            return SimpleNamespace(
                to_dict=lambda: {"stable_episode_id": stable_episode_id}
            )

        def close(self):
            self.closed = True

    env = FakeEnv()
    monkeypatch.setattr(
        "scripts.materialize_lift_anything_episode_specs.gym.make",
        lambda *args, **kwargs: env,
    )

    result = _materialize(
        object_spec_dict=ObjectSpec(
            source="cube",
            object_id="retry-cube",
            cube_half_size=0.02,
            cube_color=(1.0, 0.0, 0.0, 1.0),
        ).to_dict(),
        stable_episode_id="cube-retry-0001",
        environment_seed_start=100,
        max_attempts=2,
        render_backend="cuda:0",
    )

    assert env.seeds == [100, 101]
    assert env.closed
    assert result["episode"]["stable_episode_id"] == "cube-retry-0001"
    assert [row["status"] for row in result["attempts"]] == [
        "rejected",
        "accepted",
    ]


def test_materialization_coordinates_preserve_global_shard_identity() -> None:
    spec = ObjectSpec(
        source="cube",
        object_id="sharded-cube",
        cube_half_size=0.02,
        cube_color=(1.0, 0.0, 0.0, 1.0),
    )

    stable_episode_id, seed = materialization_coordinates(
        spec,
        local_index=3,
        episode_index_offset=64,
        seed_start=200_000,
        max_attempts=16,
    )

    assert stable_episode_id == "cube-000067"
    assert seed == 201_072


def test_quick_materialization_sample_is_deterministic_and_unique() -> None:
    specs = [
        ObjectSpec(
            source="interndata",
            object_id=f"object-{index:03d}",
            category="test",
        )
        for index in range(20)
    ]

    first = select_object_specs(specs, sample_count=10, seed=17)
    second = select_object_specs(list(reversed(specs)), sample_count=10, seed=17)

    assert [spec.stable_id for spec in first] == [spec.stable_id for spec in second]
    assert len({spec.stable_id for spec in first}) == 10


def test_supplement_prefers_unique_failed_interndata_objects() -> None:
    base = _episode_spec()
    episodes = [
        dataclasses.replace(
            base,
            stable_episode_id=f"interdata-{index:06d}",
            object_spec=ObjectSpec(
                source="interndata",
                object_id=object_id,
                category="category",
            ),
        )
        for index, object_id in enumerate(("a", "a", "b"))
    ]
    results = [
        {"stable_episode_id": episode.stable_episode_id, "accepted": False}
        for episode in episodes
    ]

    specs, metadata = build_supplemental_specs(
        episodes,
        results,
        source_counts={"interndata": 2},
    )

    assert [spec.object_id for spec in specs] == ["a", "b"]
    assert metadata["interndata"]["selection_pool"] == 2


def test_supplement_can_select_accepted_or_all_results() -> None:
    base = _episode_spec()
    episodes = [
        dataclasses.replace(
            base,
            stable_episode_id=f"ycb-{index:06d}",
            object_spec=ObjectSpec(source="ycb", object_id=object_id),
        )
        for index, object_id in enumerate(("a", "a", "b"))
    ]
    results = [
        {"stable_episode_id": episodes[0].stable_episode_id, "accepted": True},
        {"stable_episode_id": episodes[1].stable_episode_id, "accepted": False},
        {"stable_episode_id": episodes[2].stable_episode_id, "accepted": False},
    ]

    accepted, _ = build_supplemental_specs(
        episodes,
        results,
        source_counts={"ycb": 2},
        source_status={"ycb": "accepted"},
    )
    balanced, _ = build_supplemental_specs(
        episodes,
        results,
        source_counts={"ycb": 3},
        source_status={"ycb": "all"},
        unique_sources={"ycb"},
    )

    assert [spec.object_id for spec in accepted] == ["a", "a"]
    assert [spec.object_id for spec in balanced] == ["a", "b", "a"]


def test_huggingface_repo_types_are_importable_from_supported_module() -> None:
    from huggingface_hub.hf_api import RepoFile, RepoFolder

    assert RepoFile.__module__ == "huggingface_hub.hf_api"
    assert RepoFolder.__module__ == "huggingface_hub.hf_api"
