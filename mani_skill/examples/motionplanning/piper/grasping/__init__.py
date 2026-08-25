"""Reusable grasp-proposal and benchmark contracts for PIPER."""

from .contracts import (
    BenchmarkGroup,
    CandidateEvaluation,
    FailureStage,
    GraspCandidate,
    GraspProviderName,
    PipelineName,
)
from .antipodal import AntipodalConfig, AntipodalGraspProvider
from .geometry import ResolvedObjectGeometry, resolve_object_geometry

__all__ = [
    "BenchmarkGroup",
    "CandidateEvaluation",
    "FailureStage",
    "GraspCandidate",
    "GraspProviderName",
    "PipelineName",
    "AntipodalConfig",
    "AntipodalGraspProvider",
    "ResolvedObjectGeometry",
    "resolve_object_geometry",
]
