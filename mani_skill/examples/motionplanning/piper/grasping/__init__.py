"""Reusable grasp-proposal and benchmark contracts for PIPER."""

from .antipodal import AntipodalConfig, AntipodalGraspProvider
from .contracts import (
    BenchmarkGroup,
    CandidateEvaluation,
    FailureStage,
    GraspCandidate,
    GraspProviderName,
    PipelineName,
)
from .geometry import ResolvedObjectGeometry, resolve_object_geometry
from .graspgenx import GraspGenXConfig, GraspGenXGraspProvider
from .tabletop import TabletopConfig, TabletopGraspProvider

__all__ = [
    "AntipodalConfig",
    "AntipodalGraspProvider",
    "BenchmarkGroup",
    "CandidateEvaluation",
    "FailureStage",
    "GraspCandidate",
    "GraspGenXConfig",
    "GraspGenXGraspProvider",
    "GraspProviderName",
    "PipelineName",
    "ResolvedObjectGeometry",
    "TabletopConfig",
    "TabletopGraspProvider",
    "resolve_object_geometry",
]
