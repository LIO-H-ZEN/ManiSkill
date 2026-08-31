import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mani_skill.examples.motionplanning.piper.grasping.contracts import (
    CandidateEvaluation,
    GraspProviderName,
    PipelineName,
)
from mani_skill.examples.motionplanning.piper.solutions.lift_anything import (
    ExpertResult,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "collect_robodojo_general_pickup_piper.py"
)
SPEC = importlib.util.spec_from_file_location("collect_general_pickup", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
collect = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collect
SPEC.loader.exec_module(collect)


def test_parse_layout_ids_is_strict() -> None:
    assert collect.parse_layout_ids("9,2") == (9, 2)
    with pytest.raises(ValueError, match="non-empty"):
        collect.parse_layout_ids("")
    with pytest.raises(ValueError, match="unique"):
        collect.parse_layout_ids("9,9")
    with pytest.raises(ValueError, match="non-negative"):
        collect.parse_layout_ids("-1")


@pytest.mark.parametrize(
    ("action", "expected"),
    [(-1.0, 0.0), (0.0, 0.0175), (1.0, 0.035)],
)
def test_gripper_action_to_joint7(action: float, expected: float) -> None:
    assert collect.gripper_action_to_joint7(action) == pytest.approx(expected)


def test_gripper_action_to_joint7_fast_fails() -> None:
    for value in (-1.01, 1.01, np.nan):
        with pytest.raises(ValueError, match="gripper action"):
            collect.gripper_action_to_joint7(value)


def test_write_episode_preserves_dynamic_prompt_and_contract(tmp_path) -> None:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    frame = {
        "images": {name: image.copy() for name in collect.CAMERA_UIDS},
        "state": np.zeros(7, dtype=np.float32),
        "actual_qpos": np.zeros(8, dtype=np.float32),
        "action": np.zeros(7, dtype=np.float32),
        "expert_stage": "pregrasp",
    }
    prompt = "Pick up the white toy car by 10 cm."
    layout_spec = {"contract_sha256": "a" * 64, "target": {"category": "toy_car"}}

    path, digest = collect._write_episode(
        output_dir=tmp_path,
        episode_id="episode-009",
        frames=[frame],
        prompt=prompt,
        layout_spec=layout_spec,
        metadata={"robust_success_10step": True},
    )

    assert collect._sha256_file(path) == digest
    with np.load(path, allow_pickle=False) as payload:
        assert payload["prompt"].item() == prompt
        assert payload["observation.state"].shape == (1, 7)
        assert payload["observation.actual_qpos"].shape == (1, 8)
        assert payload["action_command_raw"].shape == (1, 7)
        assert payload["layout_spec_json"].shape == ()


def test_solve_provider_request_routes_cascade_with_independent_budget(
    monkeypatch,
) -> None:
    expected = ExpertResult(False, "exhausted", None, 0, None)
    calls = []

    def fake_cascade(env, **kwargs):
        calls.append((env, kwargs))
        return expected

    monkeypatch.setattr(collect, "solve_provider_cascade", fake_cascade)
    env = object()
    result = collect.solve_provider_request(
        env,
        provider_request="cascade",
        seed=11,
        grasp_cache_dir=None,
        candidate_pool_limit=64,
    )

    assert result is expected
    assert calls == [
        (
            env,
            {
                "seed": 11,
                "grasp_cache_dir": None,
                "candidate_pool_limit": 64,
            },
        )
    ]


def test_solve_provider_request_preserves_single_provider_entrypoint(
    monkeypatch,
) -> None:
    expected = ExpertResult(False, "failed", None, 0, None)
    calls = []

    def fake_solve(env, **kwargs):
        calls.append((env, kwargs))
        return expected

    monkeypatch.setattr(collect, "solve", fake_solve)
    env = object()
    result = collect.solve_provider_request(
        env,
        provider_request="tabletop",
        seed=3,
        grasp_cache_dir=Path("cache"),
        candidate_pool_limit=32,
    )

    assert result is expected
    assert calls[0][0] is env
    assert calls[0][1]["provider"] is GraspProviderName.TABLETOP
    assert calls[0][1]["pipeline"] == "common"


def test_selected_provider_requires_successful_candidate_evaluation() -> None:
    evaluation = CandidateEvaluation(
        candidate_id="pair-1",
        provider=GraspProviderName.ANTIPODAL,
        pipeline=PipelineName.COMMON,
        rank=1,
        failure_stage=None,
        failure_reason=None,
        robust_success_10step=True,
    )
    result = ExpertResult(
        True,
        "accepted",
        "pair-1",
        1,
        (None, None, None, None, {}),
        (evaluation,),
    )

    assert collect.selected_provider(result) == "antipodal"

    invalid = ExpertResult(
        True,
        "accepted",
        "pair-1",
        1,
        (None, None, None, None, {}),
    )
    with pytest.raises(RuntimeError, match="unique successful provider"):
        collect.selected_provider(invalid)


def test_maximum_attempt_lift_height_includes_previous_provider_attempts() -> None:
    evaluation = CandidateEvaluation(
        candidate_id="tabletop-candidate",
        provider=GraspProviderName.TABLETOP,
        pipeline=PipelineName.COMMON,
        rank=1,
        failure_stage=None,
        failure_reason=None,
        max_lift_height=0.074,
    )
    result = ExpertResult(
        False,
        "provider-cascade-exhausted",
        None,
        1,
        None,
        (evaluation,),
    )

    assert collect.maximum_attempt_lift_height(result, 0.0) == pytest.approx(0.074)


def test_maximum_attempt_lift_height_rejects_invalid_values() -> None:
    result = ExpertResult(False, "failed", None, 0, None)

    with pytest.raises(ValueError, match="finite and non-negative"):
        collect.maximum_attempt_lift_height(result, np.nan)


def test_collect_layout_validates_self_collision_contract(tmp_path) -> None:
    with pytest.raises(ValueError, match="disable_self_collisions must be a boolean"):
        collect.collect_layout(
            layout_id=9,
            contract_root=tmp_path,
            asset_root=tmp_path,
            grasp_cache_dir=tmp_path,
            output_dir=tmp_path,
            render_backend="cuda:0",
            candidate_pool_limit=32,
            seed=0,
            disable_self_collisions="yes",
        )


def test_collect_layout_applies_self_collision_contract_before_env_creation(
    monkeypatch, tmp_path
) -> None:
    contract_root = tmp_path / "contracts"
    contract_root.mkdir()
    (contract_root / "general_pickup_filtered_000.json").write_text("{}")

    def reject_after_contract(*args, **kwargs):
        assert collect.PiperWristCam.disable_self_collisions is True
        raise RuntimeError("stop after self-collision contract")

    monkeypatch.setattr(collect.gym, "make", reject_after_contract)

    with pytest.raises(RuntimeError, match="stop after self-collision contract"):
        collect.collect_layout(
            layout_id=0,
            contract_root=contract_root,
            asset_root=tmp_path / "assets",
            grasp_cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
            render_backend="cuda:0",
            candidate_pool_limit=32,
            seed=0,
            disable_self_collisions=True,
        )
