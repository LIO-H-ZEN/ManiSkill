import json

import gymnasium as gym
import numpy as np
import pytest

from mani_skill.envs.tasks.pick_anything.episode_specs import (
    EpisodeSpec,
    ObjectSpec,
    SettledObjectState,
)
from mani_skill.examples.motionplanning.piper.grasping.benchmark import (
    analyze_benchmark,
)
from scripts.freeze_lift_anything_benchmark import (
    BENCHMARK_CATEGORIES,
    freeze_manifest,
    freeze_quick_manifest,
)
from scripts.collect_lift_anything_piper import AttemptResult, MetricsEpisodeRecorder
from scripts import run_lift_anything_grasp_benchmark as benchmark_runner


def _episode(object_id: str, episode_index: int) -> EpisodeSpec:
    return EpisodeSpec(
        stable_episode_id=f"{object_id}-{episode_index}",
        environment_seed=episode_index,
        object_spec=ObjectSpec(
            source="cube",
            object_id=object_id,
            cube_half_size=0.02,
            cube_color=(1.0, 0.0, 0.0, 1.0),
        ),
        object_position=(0.0, 0.0, 0.02),
        object_quaternion=(1.0, 0.0, 0.0, 0.0),
        table_kind="wood",
        table_texture=None,
        table_friction=None,
        floor_texture=None,
        hdri=None,
        directional_light_direction=(1.0, 0.0, -1.0),
        directional_light_intensity=1.0,
        robot_init_qpos=(0.0,) * 8,
        settled_object_state=SettledObjectState(
            position=(0.0, 0.0, 0.02),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        ),
    )


def test_freeze_manifest_is_stratified_and_hashes_frozen_contract() -> None:
    categories = {f"cube/{category}": category for category in BENCHMARK_CATEGORIES}
    episodes = [
        _episode(category, pose)
        for category in BENCHMARK_CATEGORIES
        for pose in range(2)
    ]

    manifest = freeze_manifest(
        episodes,
        categories,
        split="pilot",
        objects_per_category=1,
        poses_per_object=2,
        seed=3,
    )

    assert len(manifest["episodes"]) == 8
    assert len(manifest["benchmark_manifest_sha256"]) == 64
    assert set(manifest["object_categories"].values()) == set(BENCHMARK_CATEGORIES)
    assert "antipodal_config_fingerprint" in manifest["frozen_versions"]
    assert "collision_model_fingerprint" in manifest["frozen_versions"]


def test_benchmark_analysis_uses_object_level_paired_bootstrap() -> None:
    episode_to_object = {"simple-0": "simple", "thin-0": "thin"}
    categories = {"simple": "simple_convex", "thin": "thin_flat"}
    failed = {
        "accepted": False,
        "attempted_candidates": 16,
        "reason": "lift",
        "elapsed_seconds": 2.0,
        "candidate_evaluations": (
            {
                "rank": 1,
                "geometry_feasible": True,
                "ik_feasible": False,
                "path_feasible": False,
                "failure_stage": "ik",
            },
        ),
    }
    succeeded = {
        "accepted": True,
        "attempted_candidates": 1,
        "reason": "accepted",
        "elapsed_seconds": 1.0,
        "candidate_evaluations": (
            {
                "rank": 1,
                "geometry_feasible": True,
                "ik_feasible": True,
                "path_feasible": True,
                "failure_stage": None,
            },
        ),
    }
    group_a = [
        {"stable_episode_id": episode_id, **failed} for episode_id in episode_to_object
    ]
    group_b = [
        {"stable_episode_id": episode_id, **succeeded}
        for episode_id in episode_to_object
    ]

    report = analyze_benchmark(
        {"A": group_a, "B": group_b},
        episode_to_object,
        categories,
        bootstrap_samples=100,
        seed=2,
    )

    assert report["groups"]["B"]["success_at_k"]["1"] == 1.0
    assert report["groups"]["A"]["geometry_feasible_recall_at_k"]["1"] == 1.0
    assert report["groups"]["B"]["ik_feasible_recall_at_k"]["1"] == 1.0
    assert report["groups"]["A"]["candidate_failure_stage_distribution"] == {"ik": 2}
    assert report["groups"]["B"]["successful_candidate_rank"]["median"] == 1.0
    assert report["groups"]["B"]["execution_seconds"]["p95"] == 1.0
    assert report["groups"]["B"]["grasp_oracle_status"] == "not_run"
    assert report["groups"]["B"]["grasp_oracle_recall_at_k"]["16"] is None
    assert report["paired_antipodal_minus_obb"]["promotion_passed"]


def test_quick_benchmark_reports_delta_without_promotion_decision() -> None:
    episode_to_object = {"object-0": "object"}
    categories = {"object": "unstratified"}
    row = {
        "stable_episode_id": "object-0",
        "accepted": True,
        "attempted_candidates": 1,
        "elapsed_seconds": 1.0,
        "reason": "accepted",
        "candidate_evaluations": (),
    }

    report = analyze_benchmark(
        {"A": [row], "B": [row]},
        episode_to_object,
        categories,
        bootstrap_samples=10,
        evaluate_promotion=False,
    )

    comparison = report["paired_antipodal_minus_obb"]
    assert comparison["mean"] == 0.0
    assert comparison["promotion_evaluated"] is False
    assert comparison["promotion_passed"] is None


def test_formal_manifest_can_exclude_pilot_objects() -> None:
    categories = {
        f"cube/{category}-{index}": category
        for category in BENCHMARK_CATEGORIES
        for index in range(2)
    }
    episodes = [
        _episode(f"{category}-{index}", 0)
        for category in BENCHMARK_CATEGORIES
        for index in range(2)
    ]
    excluded = {f"cube/{category}-0" for category in BENCHMARK_CATEGORIES}

    manifest = freeze_manifest(
        episodes,
        categories,
        split="formal",
        objects_per_category=1,
        poses_per_object=1,
        seed=5,
        excluded_objects=excluded,
    )

    assert set(manifest["object_categories"]).isdisjoint(excluded)


def test_quick_manifest_accepts_one_layout_for_100_unstratified_objects() -> None:
    episodes = [_episode(f"object-{index:03d}", 0) for index in range(100)]

    manifest = freeze_quick_manifest(episodes, episode_count=100, seed=20260825)

    assert manifest["split"] == "quick"
    assert manifest["promotion_eligible"] is False
    assert len(manifest["episodes"]) == 100
    assert set(manifest["object_categories"].values()) == {"unstratified"}


class _MetricsEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(1, dtype=np.float32),
            0.0,
            False,
            False,
            {"lift_height": np.asarray([0.12], dtype=np.float32)},
        )


def test_metrics_episode_recorder_tracks_steps_without_rgb_frames() -> None:
    recorder = MetricsEpisodeRecorder(_MetricsEnv())

    recorder.reset(seed=1)
    recorder.step(np.zeros(7, dtype=np.float32))

    assert recorder.step_count == 1
    assert np.isclose(recorder.max_lift_height, 0.12)
    assert not hasattr(recorder, "frames")


def test_benchmark_episode_result_is_written_as_resumable_shard(
    tmp_path, monkeypatch
) -> None:
    episode = _episode("simple", 0)
    expected = AttemptResult(
        stable_episode_id=episode.stable_episode_id,
        object_id=episode.object_spec.stable_id,
        source=episode.object_spec.source,
        accepted=True,
        reason="accepted",
        attempted_candidates=1,
        steps=10,
        max_lift_height=0.12,
        shard_path=None,
        shard_sha256=None,
        provider="obb",
        pipeline="common",
    )
    monkeypatch.setattr(benchmark_runner, "collect_attempt", lambda **kwargs: expected)
    result_path = tmp_path / "results" / f"{episode.stable_episode_id}.json"

    result = benchmark_runner._run_one(
        group="A",
        episode_dict=episode.to_dict(),
        group_dir=str(tmp_path),
        result_path=str(result_path),
        render_backend="cuda:0",
        cache_dir=str(tmp_path / "cache"),
    )

    assert result == json.loads(result_path.read_text())
    assert result["stable_episode_id"] == episode.stable_episode_id


def test_benchmark_rejects_stale_resumable_shard() -> None:
    episode = _episode("simple", 0)

    with pytest.raises(ValueError, match="stale or mismatched"):
        benchmark_runner._validate_completed_result(
            {
                "stable_episode_id": episode.stable_episode_id,
                "object_id": episode.object_spec.stable_id,
                "benchmark_group": "A",
                "provider": "obb",
                "pipeline": "common",
                "episode_spec_fingerprint": "stale",
            },
            episode,
            benchmark_runner.BenchmarkGroup.A,
        )
