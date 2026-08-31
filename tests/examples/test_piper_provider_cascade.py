from types import SimpleNamespace

import pytest

from mani_skill.examples.motionplanning.piper.grasping.contracts import (
    CandidateEvaluation,
    FailureStage,
    GraspProviderName,
    PipelineName,
)
from mani_skill.examples.motionplanning.piper.solutions import lift_anything


def _result(
    *,
    provider: GraspProviderName,
    success: bool,
    attempted: int,
) -> lift_anything.ExpertResult:
    evaluation = CandidateEvaluation(
        candidate_id=f"{provider.value}-candidate",
        provider=provider,
        pipeline=PipelineName.COMMON,
        rank=1,
        failure_stage=None if success else FailureStage.GRIPPER_COLLISION,
        failure_reason=None if success else "rejected",
        robust_success_10step=success,
    )
    return lift_anything.ExpertResult(
        success=success,
        reason="accepted" if success else "no-feasible-candidate",
        candidate_id=evaluation.candidate_id if success else None,
        attempted_candidates=attempted,
        transition=(None, None, None, None, {}) if success else None,
        evaluations=(evaluation,),
    )


def test_provider_cascade_keeps_independent_budgets_and_stops_on_success(
    monkeypatch,
) -> None:
    calls = []
    results = {
        GraspProviderName.TABLETOP: _result(
            provider=GraspProviderName.TABLETOP, success=False, attempted=3
        ),
        GraspProviderName.ANTIPODAL: _result(
            provider=GraspProviderName.ANTIPODAL, success=True, attempted=2
        ),
    }

    def fake_solve(env, **kwargs):
        del env
        calls.append(kwargs)
        return results[kwargs["provider"]]

    monkeypatch.setattr(lift_anything, "solve", fake_solve)

    result = lift_anything.solve_provider_cascade(
        SimpleNamespace(),
        seed=7,
        providers=(GraspProviderName.TABLETOP, GraspProviderName.ANTIPODAL),
        grasp_cache_dir=None,
        candidate_pool_limit=64,
    )

    assert [call["provider"] for call in calls] == [
        GraspProviderName.TABLETOP,
        GraspProviderName.ANTIPODAL,
    ]
    assert all(call["candidate_pool_limit"] == 64 for call in calls)
    assert result.success
    assert result.candidate_id == "antipodal-candidate"
    assert result.attempted_candidates == 5
    assert [item.provider for item in result.evaluations] == [
        GraspProviderName.TABLETOP,
        GraspProviderName.ANTIPODAL,
    ]


def test_default_cascade_preserves_baseline_before_offset_provider() -> None:
    assert lift_anything.DEFAULT_GRASP_PROVIDER_CASCADE == (
        GraspProviderName.TABLETOP,
        GraspProviderName.TABLETOP_OFFSET,
        GraspProviderName.ANTIPODAL,
        GraspProviderName.OBB,
    )


def test_provider_cascade_reports_all_failures(monkeypatch) -> None:
    providers = (GraspProviderName.TABLETOP, GraspProviderName.OBB)

    def fake_solve(env, **kwargs):
        del env
        return _result(provider=kwargs["provider"], success=False, attempted=1)

    monkeypatch.setattr(lift_anything, "solve", fake_solve)

    result = lift_anything.solve_provider_cascade(
        SimpleNamespace(),
        seed=0,
        providers=providers,
        grasp_cache_dir=None,
        candidate_pool_limit=32,
    )

    assert not result.success
    assert result.reason == (
        "provider-cascade-exhausted: "
        "tabletop=no-feasible-candidate; obb=no-feasible-candidate"
    )
    assert result.attempted_candidates == 2
    assert len(result.evaluations) == 2


def test_provider_cascade_reports_expected_provider_unavailability(monkeypatch) -> None:
    def fake_solve(env, **kwargs):
        del env, kwargs
        raise RuntimeError("width-infeasible: object is wider than the gripper")

    monkeypatch.setattr(lift_anything, "solve", fake_solve)

    result = lift_anything.solve_provider_cascade(
        SimpleNamespace(),
        seed=0,
        providers=(GraspProviderName.OBB,),
        grasp_cache_dir=None,
        candidate_pool_limit=32,
    )

    assert not result.success
    assert result.reason == (
        "provider-cascade-exhausted: "
        "obb=width-infeasible: object is wider than the gripper"
    )


def test_provider_cascade_does_not_hide_unexpected_provider_errors(monkeypatch) -> None:
    def fake_solve(env, **kwargs):
        del env, kwargs
        raise RuntimeError("broken planner invariant")

    monkeypatch.setattr(lift_anything, "solve", fake_solve)

    with pytest.raises(RuntimeError, match="broken planner invariant"):
        lift_anything.solve_provider_cascade(
            SimpleNamespace(),
            seed=0,
            providers=(GraspProviderName.OBB,),
            grasp_cache_dir=None,
            candidate_pool_limit=32,
        )


@pytest.mark.parametrize(
    ("providers", "message"),
    [
        ((), "at least one"),
        (
            (GraspProviderName.TABLETOP, GraspProviderName.TABLETOP),
            "duplicate",
        ),
    ],
)
def test_provider_cascade_rejects_invalid_provider_order(
    providers,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        lift_anything.solve_provider_cascade(
            SimpleNamespace(),
            seed=0,
            providers=providers,
            grasp_cache_dir=None,
            candidate_pool_limit=64,
        )
