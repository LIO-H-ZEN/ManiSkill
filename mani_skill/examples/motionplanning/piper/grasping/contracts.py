"""Versioned contracts shared by grasp providers and benchmark pipelines."""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Mapping

import numpy as np

FRAME_CONVENTION_VERSION = "piper_object_tcp_v1"


class GraspProviderName(str, Enum):
    OBB = "obb"
    ANTIPODAL = "antipodal"


class PipelineName(str, Enum):
    LEGACY = "legacy"
    COMMON = "common"


class BenchmarkGroup(str, Enum):
    L0 = "L0"
    A = "A"
    B = "B"

    @property
    def provider(self) -> GraspProviderName:
        return (
            GraspProviderName.ANTIPODAL
            if self is BenchmarkGroup.B
            else GraspProviderName.OBB
        )

    @property
    def pipeline(self) -> PipelineName:
        return PipelineName.LEGACY if self is BenchmarkGroup.L0 else PipelineName.COMMON


class FailureStage(str, Enum):
    PROPOSAL = "proposal"
    GEOMETRY = "geometry"
    TABLE_CLEARANCE = "table_clearance"
    GRIPPER_COLLISION = "gripper_collision"
    IK = "ik"
    PREGRASP_PATH = "pregrasp_path"
    DESCEND_PATH = "descend_path"
    CLOSE = "close"
    LIFT = "lift"
    ROBUST_HOLD = "robust_hold"
    TIMEOUT = "timeout"


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    return array.copy()


@dataclasses.dataclass(frozen=True)
class GraspCandidate:
    """A provider proposal expressed in the SAPIEN actor-local frame.

    ``object_T_tcp`` maps the ManiSkill ``piper_tcp`` frame into the actor
    frame. Its rotation columns follow ``[ortho, closing, approach]``.
    """

    candidate_id: str
    source: GraspProviderName
    object_T_tcp: np.ndarray
    required_width: float
    proposal_score: float
    contact_points: np.ndarray
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        object.__setattr__(self, "source", GraspProviderName(self.source))
        transform = _finite_array(self.object_T_tcp, (4, 4), "object_T_tcp")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError("object_T_tcp must be a homogeneous transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("object_T_tcp rotation must be orthonormal")
        if np.linalg.det(rotation) < 0.99999:
            raise ValueError("object_T_tcp rotation must be right-handed")
        object.__setattr__(self, "object_T_tcp", transform)
        if not 0.001 <= float(self.required_width) <= 0.068:
            raise ValueError("required_width must be in [0.001, 0.068] m")
        if not np.isfinite(self.proposal_score):
            raise ValueError("proposal_score must be finite")
        contacts = _finite_array(self.contact_points, (2, 3), "contact_points")
        object.__setattr__(self, "contact_points", contacts)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclasses.dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    provider: GraspProviderName
    pipeline: PipelineName
    rank: int
    failure_stage: FailureStage | None
    failure_reason: str | None
    antipodal_score: float | None = None
    clearance_score: float | None = None
    ik_cost: float | None = None
    path_length: float | None = None
    com_distance: float | None = None
    gravity_torque_risk: float | None = None
    max_lift_height: float = 0.0
    legacy_success_3step: bool = False
    robust_success_10step: bool = False
    geometry_feasible: bool = False
    ik_feasible: bool = False
    path_feasible: bool = False
    executed: bool = False

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        object.__setattr__(self, "provider", GraspProviderName(self.provider))
        object.__setattr__(self, "pipeline", PipelineName(self.pipeline))
        if self.failure_stage is not None:
            object.__setattr__(self, "failure_stage", FailureStage(self.failure_stage))
            if not self.failure_reason:
                raise ValueError("failed candidate evaluation requires failure_reason")
        elif self.failure_reason is not None:
            raise ValueError(
                "successful candidate evaluation cannot have failure_reason"
            )
        if self.ik_feasible and not self.geometry_feasible:
            raise ValueError("IK feasibility requires geometry feasibility")
        if self.path_feasible and not self.ik_feasible:
            raise ValueError("path feasibility requires IK feasibility")
        if self.executed and not self.path_feasible:
            raise ValueError("candidate execution requires path feasibility")

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["provider"] = self.provider.value
        payload["pipeline"] = self.pipeline.value
        payload["failure_stage"] = (
            None if self.failure_stage is None else self.failure_stage.value
        )
        return payload
