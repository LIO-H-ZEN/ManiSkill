import numpy as np
import pytest

from mani_skill.examples.motionplanning.piper.grasping import (
    BenchmarkGroup,
    CandidateEvaluation,
    FailureStage,
    GraspCandidate,
    GraspProviderName,
    PipelineName,
)


def test_benchmark_groups_lock_proposal_and_pipeline_matrix() -> None:
    assert BenchmarkGroup.L0.provider is GraspProviderName.OBB
    assert BenchmarkGroup.L0.pipeline is PipelineName.LEGACY
    assert BenchmarkGroup.A.provider is GraspProviderName.OBB
    assert BenchmarkGroup.A.pipeline is PipelineName.COMMON
    assert BenchmarkGroup.B.provider is GraspProviderName.ANTIPODAL
    assert BenchmarkGroup.B.pipeline is PipelineName.COMMON


def test_object_local_grasp_candidate_validates_frame_contract() -> None:
    candidate = GraspCandidate(
        candidate_id="pair-0001-roll-00",
        source="antipodal",
        object_T_tcp=np.eye(4),
        required_width=0.02,
        proposal_score=0.8,
        contact_points=np.array([[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0]]),
    )

    assert candidate.source is GraspProviderName.ANTIPODAL
    np.testing.assert_array_equal(candidate.object_T_tcp, np.eye(4))


def test_grasp_candidate_rejects_left_handed_tcp_frame() -> None:
    transform = np.eye(4)
    transform[0, 0] = -1.0

    with pytest.raises(ValueError, match="right-handed"):
        GraspCandidate(
            candidate_id="invalid",
            source="obb",
            object_T_tcp=transform,
            required_width=0.02,
            proposal_score=0.0,
            contact_points=np.zeros((2, 3)),
        )


def test_candidate_evaluation_requires_failure_reason() -> None:
    with pytest.raises(ValueError, match="requires failure_reason"):
        CandidateEvaluation(
            candidate_id="candidate",
            provider="obb",
            pipeline="common",
            rank=1,
            failure_stage=FailureStage.IK,
            failure_reason=None,
        )
