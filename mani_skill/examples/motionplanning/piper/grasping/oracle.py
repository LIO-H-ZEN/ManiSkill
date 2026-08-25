"""Gripper-only oracle metrics, independent of arm IK and motion planning."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable

from .contracts import GraspCandidate


@dataclasses.dataclass(frozen=True)
class GraspOracleAttempt:
    candidate_id: str
    success: bool
    reason: str
    max_lift_height: float
    robust_hold_steps: int


@dataclasses.dataclass(frozen=True)
class GraspOracleResult:
    attempts: tuple[GraspOracleAttempt, ...]
    recall_at_k: dict[int, bool]


def evaluate_gripper_oracle(
    candidates: Iterable[GraspCandidate],
    evaluator: Callable[[GraspCandidate], GraspOracleAttempt],
    *,
    ks: tuple[int, ...] = (1, 4, 8, 16),
) -> GraspOracleResult:
    ordered = list(candidates)
    if not ordered:
        raise ValueError("gripper oracle requires at least one candidate")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("oracle K values must be positive")
    attempts = tuple(evaluator(candidate) for candidate in ordered[: max(ks)])
    recall = {
        k: any(attempt.success for attempt in attempts[: min(k, len(attempts))])
        for k in ks
    }
    return GraspOracleResult(attempts=attempts, recall_at_k=recall)
