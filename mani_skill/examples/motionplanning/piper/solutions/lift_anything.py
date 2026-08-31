"""Bounded OBB-based expert for LiftAnythingPiper-v1."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Iterable

import numpy as np
import sapien

from mani_skill.envs.tasks.pick_anything.lift_anything_piper import (
    LiftAnythingPiperEnv,
)
from mani_skill.examples.motionplanning.piper.grasping.antipodal import (
    ANTIPODAL_PROVIDER_VERSION,
    AntipodalConfig,
    AntipodalGraspProvider,
)
from mani_skill.examples.motionplanning.piper.grasping.cache import (
    GraspCache,
    cache_key,
    piper_gripper_geometry_hash,
)
from mani_skill.examples.motionplanning.piper.grasping.contracts import (
    CandidateEvaluation,
    FailureStage,
    GraspProviderName,
    PipelineName,
)
from mani_skill.examples.motionplanning.piper.grasping.contracts import (
    GraspCandidate as LocalGraspCandidate,
)
from mani_skill.examples.motionplanning.piper.grasping.geometry import (
    GEOMETRY_PREPROCESS_VERSION,
    resolve_object_geometry,
)
from mani_skill.examples.motionplanning.piper.grasping.graspgenx import (
    GRASPGENX_PROVIDER_VERSION,
    GraspGenXConfig,
    GraspGenXGraspProvider,
)
from mani_skill.examples.motionplanning.piper.grasping.gripper_geometry import (
    PiperGripperGeometry,
)
from mani_skill.examples.motionplanning.piper.grasping.tabletop import (
    TABLETOP_OFFSET_LATERAL_FRACTIONS,
    TabletopConfig,
    TabletopGraspProvider,
)
from mani_skill.examples.motionplanning.piper.grasping.pipeline import (
    LIFT_DISTANCE,
    ROBUST_HOLD_STEPS,
    RankedCandidate,
    TargetContactValidator,
    TargetGeometryCollisionValidator,
    dense_translation_waypoints,
    minimum_gripper_clearance,
    pose_matrix,
    pregrasp_pose,
    rank_candidates,
    scene_collision_points,
    world_grasp_pose,
)
from mani_skill.examples.motionplanning.piper.motionplanner import (
    PiperMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.piper.solutions.lift_cube import (
    FULL_TILT_PLANAR_REACH,
    MAX_APPROACH_TILT,
    UNTILTED_PLANAR_REACH,
)

MAX_GRASP_CANDIDATES = 16
MAX_GRIPPER_WIDTH = 0.068
DEFAULT_GRASP_PROVIDER_CASCADE = (
    GraspProviderName.TABLETOP,
    GraspProviderName.TABLETOP_OFFSET,
    GraspProviderName.ANTIPODAL,
    GraspProviderName.OBB,
)
EXPECTED_PROVIDER_UNAVAILABLE_PREFIXES = (
    "proposal-empty:",
    "width-infeasible:",
)


@dataclasses.dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    pose: sapien.Pose
    required_width: float


@dataclasses.dataclass(frozen=True)
class ExpertResult:
    success: bool
    reason: str
    candidate_id: str | None
    attempted_candidates: int
    transition: tuple | None
    evaluations: tuple[CandidateEvaluation, ...] = ()


def _reset_expert_episode(env, seed: int) -> None:
    base_env = env.unwrapped
    reconfigure = bool(getattr(base_env, "expert_reconfigure_each_candidate", True))
    env.reset(seed=seed, options={"reconfigure": reconfigure})


def _limit_candidate_pool_diversely(
    candidates: list[LocalGraspCandidate], limit: int
) -> list[LocalGraspCandidate]:
    """Round-robin regions, contact pairs, and approaches for smoke runs."""

    if limit <= 0:
        raise ValueError("candidate pool limit must be positive")

    def pair_key(candidate: LocalGraspCandidate) -> tuple:
        pair_index = candidate.metadata.get("pair_index")
        if pair_index is not None:
            return ("pair-index", int(pair_index))
        contacts = np.round(candidate.contact_points / 0.001).astype(np.int64)
        canonical = sorted(tuple(row) for row in contacts)
        return ("contacts", *canonical)

    order_index = {}

    def round_robin(groups: dict[tuple, list]) -> list:
        ordered_keys = sorted(
            groups,
            key=lambda key: order_index[groups[key][0].candidate_id],
        )
        selected = []
        round_index = 0
        while True:
            added = False
            for key in ordered_keys:
                group = groups[key]
                if round_index >= len(group):
                    continue
                selected.append(group[round_index])
                added = True
            if not added:
                return selected
            round_index += 1

    ordered = sorted(
        candidates,
        key=lambda item: (
            -int(np.floor(item.proposal_score / 0.05)),
            float(item.metadata.get("midpoint_to_com", np.inf)),
            item.candidate_id,
        ),
    )
    order_index.update(
        {candidate.candidate_id: index for index, candidate in enumerate(ordered)}
    )
    regions: dict[tuple[int, ...], dict[tuple, dict[tuple, list]]] = {}
    for candidate in ordered:
        region = tuple(candidate.metadata.get("contact_region", (0, 0, 0)))
        approach_bin = tuple(np.round(candidate.object_T_tcp[:3, 2] * 2.0).astype(int))
        regions.setdefault(region, {}).setdefault(pair_key(candidate), {}).setdefault(
            approach_bin, []
        ).append(candidate)

    region_sequences = {}
    for region, pairs in regions.items():
        pair_sequences = {
            key: round_robin(approaches) for key, approaches in pairs.items()
        }
        region_sequences[region] = round_robin(pair_sequences)
    return round_robin(region_sequences)[:limit]


def _adaptive_approach(
    center: np.ndarray, robot_base_position: np.ndarray
) -> np.ndarray:
    planar = center[:2] - robot_base_position[:2]
    planar_distance = np.linalg.norm(planar)
    if planar_distance <= 0.0:
        raise ValueError("Object center must not coincide with the PIPER base axis")
    radial = planar / planar_distance
    tilt_fraction = np.clip(
        (planar_distance - UNTILTED_PLANAR_REACH)
        / (FULL_TILT_PLANAR_REACH - UNTILTED_PLANAR_REACH),
        0.0,
        1.0,
    )
    tilt = MAX_APPROACH_TILT * tilt_fraction
    return np.array([radial[0] * np.sin(tilt), radial[1] * np.sin(tilt), -np.cos(tilt)])


def _orthogonal_closing(closing: np.ndarray, approach: np.ndarray) -> np.ndarray:
    projected = closing - approach * float(closing @ approach)
    norm = np.linalg.norm(projected)
    if norm < 1e-8:
        raise ValueError("Closing direction is parallel to approach direction")
    return projected / norm


def generate_grasp_candidates(
    agent,
    obj,
    robot_base_position: np.ndarray,
    *,
    max_candidates: int = MAX_GRASP_CANDIDATES,
) -> list[GraspCandidate]:
    if not 1 <= max_candidates <= MAX_GRASP_CANDIDATES:
        raise ValueError(
            f"max_candidates must be in [1, {MAX_GRASP_CANDIDATES}], got {max_candidates}"
        )
    mesh = obj.get_first_collision_mesh(to_world_frame=False)
    if mesh is None:
        raise RuntimeError("invalid-mesh: object has no collision mesh")
    bounds = np.asarray(mesh.bounding_box.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise RuntimeError("invalid-mesh: collision mesh has invalid bounds")
    extents = bounds[1] - bounds[0]
    if np.any(extents <= 0.0):
        raise RuntimeError(f"invalid-mesh: non-positive OBB extents {extents}")

    actor_matrix = np.asarray(obj.pose.to_transformation_matrix()[0].cpu())
    local_center = (bounds[0] + bounds[1]) / 2.0
    center = actor_matrix[:3, :3] @ local_center + actor_matrix[:3, 3]
    axes = [actor_matrix[:3, 0], actor_matrix[:3, 1]]
    approach = _adaptive_approach(center, np.asarray(robot_base_position))
    z_offsets = (0.0, 0.18, -0.18)
    yaw_offsets = np.deg2rad((0.0, 12.0, -12.0))
    candidates: list[GraspCandidate] = []
    feasible_axes = sorted(
        ((float(extents[index]), index) for index in range(2)), key=lambda item: item[0]
    )
    for required_width, axis_index in feasible_axes:
        if required_width > MAX_GRIPPER_WIDTH:
            continue
        base_axis = axes[axis_index]
        for z_fraction in z_offsets:
            grasp_center = center.copy()
            grasp_center[2] += z_fraction * extents[2]
            for yaw in yaw_offsets:
                cosine, sine = np.cos(yaw), np.sin(yaw)
                rotated = np.array(
                    [
                        cosine * base_axis[0] - sine * base_axis[1],
                        sine * base_axis[0] + cosine * base_axis[1],
                        base_axis[2],
                    ]
                )
                closing = _orthogonal_closing(rotated, approach)
                candidate_id = (
                    f"axis{axis_index}-z{z_fraction:+.2f}-yaw{np.rad2deg(yaw):+.0f}"
                )
                candidates.append(
                    GraspCandidate(
                        candidate_id,
                        agent.build_grasp_pose(approach, closing, grasp_center),
                        required_width,
                    )
                )
                if len(candidates) == max_candidates:
                    return candidates
    if not candidates:
        raise RuntimeError(
            f"width-infeasible: horizontal OBB widths {extents[:2].tolist()} exceed "
            f"{MAX_GRIPPER_WIDTH} m"
        )
    return candidates


def _obb_local_candidates(base_env) -> list[LocalGraspCandidate]:
    world_candidates = generate_grasp_candidates(
        base_env.agent,
        base_env.obj,
        np.asarray(base_env.agent.robot.pose.p[0].cpu()),
    )
    world_T_object = pose_matrix(base_env.obj.pose)
    object_T_world = np.linalg.inv(world_T_object)
    local_candidates = []
    for candidate in world_candidates:
        object_T_tcp = object_T_world @ pose_matrix(candidate.pose)
        center = object_T_tcp[:3, 3]
        closing = object_T_tcp[:3, 1]
        contacts = np.stack(
            [
                center - closing * candidate.required_width * 0.5,
                center + closing * candidate.required_width * 0.5,
            ]
        )
        local_candidates.append(
            LocalGraspCandidate(
                candidate_id=candidate.candidate_id,
                source=GraspProviderName.OBB,
                object_T_tcp=object_T_tcp,
                required_width=candidate.required_width,
                proposal_score=1.0 - candidate.required_width / MAX_GRIPPER_WIDTH,
                contact_points=contacts,
                metadata={
                    "contact_region": tuple(
                        int(value) for value in np.floor(center / 0.015)
                    )
                },
            )
        )
    return local_candidates


def _antipodal_candidates(
    base_env, *, seed: int, cache_dir: pathlib.Path | None
) -> list[LocalGraspCandidate]:
    if base_env.fixed_object_spec is None:
        raise ValueError("antipodal provider requires a fixed ObjectSpec")
    geometry = resolve_object_geometry(
        base_env.fixed_object_spec,
        asset_root=getattr(base_env, "robodojo_asset_root", None),
    )
    config = AntipodalConfig()
    gripper_hash = piper_gripper_geometry_hash()
    key = cache_key(
        geometry,
        provider=GraspProviderName.ANTIPODAL,
        provider_version=ANTIPODAL_PROVIDER_VERSION,
        provider_config_fingerprint=config.fingerprint,
        gripper_geometry_hash=gripper_hash,
    )
    cache = GraspCache(cache_dir) if cache_dir is not None else None
    if cache is not None:
        try:
            candidates, manifest = cache.load(key)
        except FileNotFoundError:
            pass
        else:
            if manifest["failure"] is not None:
                raise RuntimeError(manifest["failure"])
            return candidates
    generation_seed = int(key[:16], 16)
    candidates = AntipodalGraspProvider(config).generate(geometry, seed=generation_seed)
    if cache is not None:
        cache.store(
            key,
            candidates,
            manifest={
                "object_stable_id": base_env.fixed_object_spec.stable_id,
                "canonical_geometry_hash": geometry.canonical_geometry_hash,
                "geometry_preprocess_version": GEOMETRY_PREPROCESS_VERSION,
                "provider": GraspProviderName.ANTIPODAL.value,
                "provider_version": ANTIPODAL_PROVIDER_VERSION,
                "provider_config": dataclasses.asdict(config),
                "provider_config_fingerprint": config.fingerprint,
                "gripper_geometry_hash": gripper_hash,
                "rng_seed": generation_seed,
                "episode_seed_not_used_for_cached_geometry": int(seed),
            },
        )
    return candidates


def _graspgenx_candidates(
    base_env, *, cache_dir: pathlib.Path | None
) -> list[LocalGraspCandidate]:
    if base_env.fixed_object_spec is None:
        raise ValueError("GraspGen-X provider requires a fixed ObjectSpec")
    if cache_dir is None:
        raise ValueError(
            "GraspGen-X provider requires an offline grasp_cache_dir; "
            "model inference is intentionally isolated from ManiSkill"
        )
    geometry = resolve_object_geometry(
        base_env.fixed_object_spec,
        asset_root=getattr(base_env, "robodojo_asset_root", None),
    )
    config = GraspGenXConfig()
    provider = GraspGenXGraspProvider(config)
    key = cache_key(
        geometry,
        provider=GraspProviderName.GRASPGENX,
        provider_version=GRASPGENX_PROVIDER_VERSION,
        provider_config_fingerprint=provider.cache_fingerprint(geometry),
        gripper_geometry_hash=piper_gripper_geometry_hash(),
    )
    try:
        candidates, manifest = GraspCache(cache_dir).load(key)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"missing offline GraspGen-X cache entry {key}; "
            "run scripts/import_graspgenx_grasps.py first"
        ) from error
    if manifest["failure"] is not None:
        raise RuntimeError(manifest["failure"])
    return candidates


def _tabletop_candidates(base_env) -> list[LocalGraspCandidate]:
    if base_env.fixed_object_spec is None:
        raise ValueError("tabletop provider requires a fixed ObjectSpec")
    geometry = resolve_object_geometry(
        base_env.fixed_object_spec,
        asset_root=getattr(base_env, "robodojo_asset_root", None),
    )
    return TabletopGraspProvider().generate(
        geometry,
        world_T_object=pose_matrix(base_env.obj.pose),
        robot_base_position=np.asarray(base_env.agent.robot.pose.p[0].cpu()),
    )


def _tabletop_offset_candidates(base_env) -> list[LocalGraspCandidate]:
    if base_env.fixed_object_spec is None:
        raise ValueError("tabletop-offset provider requires a fixed ObjectSpec")
    geometry = resolve_object_geometry(
        base_env.fixed_object_spec,
        asset_root=getattr(base_env, "robodojo_asset_root", None),
    )
    candidates = TabletopGraspProvider(
        TabletopConfig(lateral_fractions=TABLETOP_OFFSET_LATERAL_FRACTIONS)
    ).generate(
        geometry,
        world_T_object=pose_matrix(base_env.obj.pose),
        robot_base_position=np.asarray(base_env.agent.robot.pose.p[0].cpu()),
    )
    return [
        dataclasses.replace(
            candidate,
            candidate_id=f"offset-{candidate.candidate_id}",
            source=GraspProviderName.TABLETOP_OFFSET,
            metadata={
                **candidate.metadata,
                "proposal_family": GraspProviderName.TABLETOP_OFFSET.value,
            },
        )
        for candidate in candidates
    ]


def _scalar_bool(value) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def _run_candidate(
    env,
    candidate: GraspCandidate,
    *,
    debug: bool,
    vis: bool,
):
    base_env: LiftAnythingPiperEnv = env.unwrapped
    planner = PiperMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=base_env.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    lift_qpos = base_env.agent.robot.get_qpos()[0, :6].cpu().numpy().copy()
    try:
        env.expert_stage = "pregrasp"
        result = planner.move_to_pose_with_RRTConnect(
            candidate.pose * sapien.Pose([0.0, 0.0, -0.07])
        )
        if result == -1:
            raise RuntimeError("pregrasp")
        if planner.episode_done:
            return planner.last_transition

        env.expert_stage = "descend"
        result = planner.move_to_pose_with_screw(candidate.pose)
        if result == -1:
            raise RuntimeError("descend")
        if planner.episode_done:
            return planner.last_transition

        env.expert_stage = "grasp"
        planner.close_gripper(t=6)
        if planner.episode_done:
            return planner.last_transition
        if not _scalar_bool(planner.last_transition[4]["is_grasped"]):
            raise RuntimeError("grasp")

        env.expert_stage = "lift"
        planner.move_to_joint_target(lift_qpos, steps=20)
        return planner.last_transition
    finally:
        planner.close()


def _path_length(result) -> float:
    positions = np.asarray(result["position"], dtype=np.float64)
    if positions.ndim != 2 or len(positions) == 0:
        raise ValueError("planner returned an invalid path")
    if len(positions) == 1:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def _path_end_robot_qpos(
    result,
    start_qpos: np.ndarray,
    *,
    gripper_width: float | None = None,
) -> np.ndarray:
    positions = np.asarray(result["position"], dtype=np.float64)
    qpos = np.asarray(start_qpos, dtype=np.float64).copy()
    if qpos.shape != (8,) or positions.ndim != 2 or positions.shape[1] != 6:
        raise ValueError(
            "PiPER planning expects an 8-DoF robot state and a 6-DoF arm path"
        )
    if len(positions) == 0 or not np.all(np.isfinite(positions)):
        raise ValueError("planner returned an empty or non-finite arm path")
    qpos[:6] = positions[-1]
    if gripper_width is not None:
        if not 0.001 <= gripper_width <= 0.068:
            raise ValueError("gripper_width must be in [0.001, 0.068] m")
        qpos[6] = gripper_width * 0.5
        qpos[7] = -gripper_width * 0.5
    return qpos


def _evaluate_common_candidates(
    env,
    candidates: list[LocalGraspCandidate],
    *,
    seed: int,
    debug: bool,
    vis: bool,
) -> tuple[list[RankedCandidate], list[CandidateEvaluation]]:
    base_env: LiftAnythingPiperEnv = env.unwrapped
    gripper_geometry = PiperGripperGeometry.from_package_assets()
    reuse_exact_scene = not bool(
        getattr(base_env, "expert_reconfigure_each_candidate", True)
    )
    shared_geometry_validator = None
    if reuse_exact_scene:
        _reset_expert_episode(env, seed)
        shared_geometry_validator = TargetGeometryCollisionValidator(
            base_env, gripper_geometry
        )
    preliminary = []
    evaluations = []
    for candidate in candidates:
        if not reuse_exact_scene:
            _reset_expert_episode(env, seed)
        grasp_pose = world_grasp_pose(base_env.obj.pose, candidate)
        grasp_matrix = pose_matrix(grasp_pose)
        target_geometry_validator = shared_geometry_validator
        if target_geometry_validator is None:
            target_geometry_validator = TargetGeometryCollisionValidator(
                base_env, gripper_geometry
            )
        try:
            executable_width = target_geometry_validator.validate_descend_and_closure(
                pregrasp_pose(grasp_pose),
                grasp_pose,
                contact_width=candidate.required_width,
            )
        except RuntimeError as error:
            evaluations.append(
                CandidateEvaluation(
                    candidate_id=candidate.candidate_id,
                    provider=candidate.source,
                    pipeline=PipelineName.COMMON,
                    rank=len(evaluations) + 1,
                    failure_stage=FailureStage.GRIPPER_COLLISION,
                    failure_reason=str(error),
                    antipodal_score=candidate.proposal_score,
                )
            )
            continue
        candidate = dataclasses.replace(
            candidate,
            required_width=executable_width,
            metadata={
                **candidate.metadata,
                "proposal_required_width": candidate.required_width,
                "executable_contact_width": executable_width,
            },
        )
        clearance = minimum_gripper_clearance(
            grasp_matrix,
            gripper_geometry,
            contact_width=candidate.required_width,
        )
        pregrasp_matrix = pose_matrix(pregrasp_pose(grasp_pose))
        clearance = min(
            clearance,
            minimum_gripper_clearance(
                pregrasp_matrix,
                gripper_geometry,
                contact_width=0.068,
            ),
        )
        if clearance < -0.001:
            evaluations.append(
                CandidateEvaluation(
                    candidate_id=candidate.candidate_id,
                    provider=candidate.source,
                    pipeline=PipelineName.COMMON,
                    rank=len(evaluations) + 1,
                    failure_stage=FailureStage.TABLE_CLEARANCE,
                    failure_reason=f"minimum clearance {clearance:.6f} m",
                    antipodal_score=candidate.proposal_score,
                    clearance_score=clearance,
                )
            )
            continue
        preliminary.append(
            RankedCandidate(
                candidate=candidate,
                clearance_score=clearance,
                ik_cost=0.0,
                path_length=0.0,
            )
        )
    ik_budget = rank_candidates(preliminary, maximum_candidates=64)
    feasible = []
    shared_collision_points = None
    shared_target_free_collision_points = None
    if reuse_exact_scene:
        shared_collision_points = scene_collision_points(base_env, include_target=True)
        shared_target_free_collision_points = scene_collision_points(
            base_env, include_target=False
        )
    for preliminary_rank, item in enumerate(ik_budget, start=1):
        _reset_expert_episode(env, seed)
        grasp_pose = world_grasp_pose(base_env.obj.pose, item.candidate)
        target = pregrasp_pose(grasp_pose)
        planner = PiperMotionPlanningSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=base_env.agent.robot.pose,
            visualize_target_grasp_pose=vis,
            print_env_info=False,
        )
        try:
            collision_points = shared_collision_points
            if collision_points is None:
                collision_points = scene_collision_points(base_env, include_target=True)
            planner.set_collision_point_cloud(collision_points)
            planning_target = planner._transform_pose_for_planning(target)
            start_qpos = base_env.agent.robot.get_qpos()[0].cpu().numpy()
            status, solutions = planner.planner.IK(
                np.concatenate([planning_target.p, planning_target.q]),
                start_qpos,
            )
            if status != "Success" or not solutions:
                evaluations.append(
                    CandidateEvaluation(
                        candidate_id=item.candidate.candidate_id,
                        provider=item.candidate.source,
                        pipeline=PipelineName.COMMON,
                        rank=preliminary_rank,
                        failure_stage=FailureStage.IK,
                        failure_reason=status,
                        antipodal_score=item.candidate.proposal_score,
                        clearance_score=item.clearance_score,
                    )
                )
                continue
            ik_cost = min(
                float(np.linalg.norm(np.asarray(solution)[:6] - start_qpos[:6]))
                for solution in solutions
            )
            path = planner.move_to_pose_with_RRTConnect(target, dry_run=True)
            if path == -1:
                evaluations.append(
                    CandidateEvaluation(
                        candidate_id=item.candidate.candidate_id,
                        provider=item.candidate.source,
                        pipeline=PipelineName.COMMON,
                        rank=preliminary_rank,
                        failure_stage=FailureStage.PREGRASP_PATH,
                        failure_reason="MPlib failed to plan a pregrasp path",
                        antipodal_score=item.candidate.proposal_score,
                        clearance_score=item.clearance_score,
                        ik_cost=ik_cost,
                    )
                )
                continue
            try:
                planner.validate_arm_path(path, stage="pregrasp")
            except RuntimeError as error:
                evaluations.append(
                    CandidateEvaluation(
                        candidate_id=item.candidate.candidate_id,
                        provider=item.candidate.source,
                        pipeline=PipelineName.COMMON,
                        rank=preliminary_rank,
                        failure_stage=FailureStage.PREGRASP_PATH,
                        failure_reason=str(error),
                        antipodal_score=item.candidate.proposal_score,
                        clearance_score=item.clearance_score,
                        ik_cost=ik_cost,
                    )
                )
                continue
            pregrasp_qpos = _path_end_robot_qpos(path, start_qpos)
            target_free_collision_points = shared_target_free_collision_points
            if target_free_collision_points is None:
                target_free_collision_points = scene_collision_points(
                    base_env, include_target=False
                )
            planner.set_collision_point_cloud(target_free_collision_points)
            descend_path = planner.plan_screw_from_qpos(grasp_pose, pregrasp_qpos)
            if descend_path["status"] != "Success":
                evaluations.append(
                    CandidateEvaluation(
                        candidate_id=item.candidate.candidate_id,
                        provider=item.candidate.source,
                        pipeline=PipelineName.COMMON,
                        rank=preliminary_rank,
                        failure_stage=FailureStage.DESCEND_PATH,
                        failure_reason=descend_path["status"],
                        antipodal_score=item.candidate.proposal_score,
                        clearance_score=item.clearance_score,
                        ik_cost=ik_cost,
                        path_length=_path_length(path),
                    )
                )
                continue
            try:
                planner.validate_arm_path(descend_path, stage="descend")
            except RuntimeError as error:
                evaluations.append(
                    CandidateEvaluation(
                        candidate_id=item.candidate.candidate_id,
                        provider=item.candidate.source,
                        pipeline=PipelineName.COMMON,
                        rank=preliminary_rank,
                        failure_stage=FailureStage.DESCEND_PATH,
                        failure_reason=str(error),
                        antipodal_score=item.candidate.proposal_score,
                        clearance_score=item.clearance_score,
                        ik_cost=ik_cost,
                        path_length=_path_length(path),
                    )
                )
                continue
            grasp_qpos = _path_end_robot_qpos(
                descend_path,
                pregrasp_qpos,
                gripper_width=item.candidate.required_width,
            )
            lift_matrix = pose_matrix(grasp_pose)
            lift_matrix[2, 3] += LIFT_DISTANCE
            lift_path = planner.plan_screw_from_qpos(
                sapien.Pose(lift_matrix), grasp_qpos
            )
            if lift_path["status"] != "Success":
                evaluations.append(
                    CandidateEvaluation(
                        candidate_id=item.candidate.candidate_id,
                        provider=item.candidate.source,
                        pipeline=PipelineName.COMMON,
                        rank=preliminary_rank,
                        failure_stage=FailureStage.LIFT,
                        failure_reason=lift_path["status"],
                        antipodal_score=item.candidate.proposal_score,
                        clearance_score=item.clearance_score,
                        ik_cost=ik_cost,
                        path_length=(_path_length(path) + _path_length(descend_path)),
                    )
                )
                continue
            try:
                planner.validate_arm_path(lift_path, stage="lift")
            except RuntimeError as error:
                evaluations.append(
                    CandidateEvaluation(
                        candidate_id=item.candidate.candidate_id,
                        provider=item.candidate.source,
                        pipeline=PipelineName.COMMON,
                        rank=preliminary_rank,
                        failure_stage=FailureStage.LIFT,
                        failure_reason=str(error),
                        antipodal_score=item.candidate.proposal_score,
                        clearance_score=item.clearance_score,
                        ik_cost=ik_cost,
                        path_length=(_path_length(path) + _path_length(descend_path)),
                    )
                )
                continue
            feasible.append(
                RankedCandidate(
                    candidate=item.candidate,
                    clearance_score=item.clearance_score,
                    ik_cost=ik_cost,
                    path_length=(
                        _path_length(path)
                        + _path_length(descend_path)
                        + _path_length(lift_path)
                    ),
                )
            )
        finally:
            planner.close()
    return rank_candidates(feasible, maximum_candidates=32), evaluations


def _run_common_candidate(
    env,
    candidate: LocalGraspCandidate,
    *,
    debug: bool,
    vis: bool,
):
    base_env: LiftAnythingPiperEnv = env.unwrapped
    planner = PiperMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=base_env.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    gripper_geometry = PiperGripperGeometry.from_package_assets()
    geometry_validator = TargetGeometryCollisionValidator(base_env, gripper_geometry)
    validator = TargetContactValidator(base_env, gripper_geometry)
    grasp_pose = world_grasp_pose(base_env.obj.pose, candidate)
    pregrasp = pregrasp_pose(grasp_pose)
    geometry_validator.validate_descend_and_closure(
        pregrasp,
        grasp_pose,
        contact_width=candidate.required_width,
    )
    grasp_matrix = pose_matrix(grasp_pose)
    for position in dense_translation_waypoints(pregrasp.p, grasp_pose.p):
        waypoint = grasp_matrix.copy()
        waypoint[:3, 3] = position
        if (
            minimum_gripper_clearance(
                waypoint,
                gripper_geometry,
                contact_width=0.068,
            )
            < -0.001
        ):
            raise RuntimeError("descend-table-collision")
    try:
        planner.set_collision_point_cloud(
            scene_collision_points(base_env, include_target=True)
        )
        env.expert_stage = "pregrasp"
        planner.execution_stage = "pregrasp"
        result = planner.move_to_pose_with_RRTConnect(pregrasp)
        if result == -1:
            raise RuntimeError("pregrasp")
        if planner.episode_done:
            return planner.last_transition

        planner.set_collision_point_cloud(
            scene_collision_points(base_env, include_target=False)
        )
        planner.waypoint_validator = lambda *, stage, progress: validator.validate(
            allow_pad_contact=True,
            progress=progress if stage == "descend" else 1.0,
        )
        env.expert_stage = "descend"
        planner.execution_stage = "descend"
        result = planner.move_to_pose_with_screw(grasp_pose)
        if result == -1:
            raise RuntimeError("descend")
        if planner.episode_done:
            return planner.last_transition

        env.expert_stage = "grasp"
        planner.execution_stage = "close"
        planner.close_gripper(t=8)
        if planner.episode_done:
            return planner.last_transition
        if not _scalar_bool(planner.last_transition[4]["is_grasped"]):
            raise RuntimeError("grasp")

        env.expert_stage = "lift"
        planner.execution_stage = "lift"
        lift_matrix = pose_matrix(grasp_pose)
        lift_matrix[2, 3] += LIFT_DISTANCE
        result = planner.move_to_pose_with_screw(sapien.Pose(lift_matrix))
        if result == -1:
            raise RuntimeError("lift")
        if not planner.episode_done:
            env.expert_stage = "hold"
            planner.execution_stage = "hold"
            planner.hold_current(ROBUST_HOLD_STEPS)
        return planner.last_transition
    finally:
        planner.close()


def solve(
    env,
    *,
    seed: int,
    candidates: Iterable[GraspCandidate] | None = None,
    provider: GraspProviderName | str = GraspProviderName.OBB,
    pipeline: PipelineName | str = PipelineName.LEGACY,
    grasp_cache_dir: pathlib.Path | None = None,
    candidate_pool_limit: int | None = None,
    debug: bool = False,
    vis: bool = False,
) -> ExpertResult:
    base_env: LiftAnythingPiperEnv = env.unwrapped
    provider = GraspProviderName(provider)
    pipeline = PipelineName(pipeline)
    if base_env.num_envs != 1:
        raise ValueError("LiftAnything expert requires num_envs=1")
    if candidate_pool_limit is not None and candidate_pool_limit < 1:
        raise ValueError("candidate_pool_limit must be positive or None")
    if candidates is not None and (
        provider is not GraspProviderName.OBB or pipeline is not PipelineName.LEGACY
    ):
        raise ValueError("explicit world-space candidates are legacy-only")
    _reset_expert_episode(env, seed)
    if pipeline is PipelineName.COMMON:
        if base_env.success_streak_steps < ROBUST_HOLD_STEPS:
            raise ValueError(
                f"common pipeline requires success_streak_steps >= {ROBUST_HOLD_STEPS}"
            )
        try:
            if provider is GraspProviderName.OBB:
                local_candidates = _obb_local_candidates(base_env)
            elif provider is GraspProviderName.ANTIPODAL:
                local_candidates = _antipodal_candidates(
                    base_env, seed=seed, cache_dir=grasp_cache_dir
                )
            elif provider is GraspProviderName.TABLETOP:
                local_candidates = _tabletop_candidates(base_env)
            elif provider is GraspProviderName.TABLETOP_OFFSET:
                local_candidates = _tabletop_offset_candidates(base_env)
            else:
                local_candidates = _graspgenx_candidates(
                    base_env, cache_dir=grasp_cache_dir
                )
        except RuntimeError as error:
            if not str(error).startswith("proposal-empty:"):
                raise
            return ExpertResult(False, str(error), None, 0, None, ())
        if candidate_pool_limit is not None:
            local_candidates = _limit_candidate_pool_diversely(
                local_candidates, candidate_pool_limit
            )
        ranked, evaluations = _evaluate_common_candidates(
            env,
            local_candidates,
            seed=seed,
            debug=debug,
            vis=vis,
        )
        if not ranked:
            return ExpertResult(
                False,
                "no-feasible-candidate",
                None,
                0,
                None,
                tuple(evaluations),
            )
        last_reason = "no_transition"
        final_transition = None
        for execution_rank, item in enumerate(ranked[:MAX_GRASP_CANDIDATES], start=1):
            _reset_expert_episode(env, seed)
            try:
                transition = _run_common_candidate(
                    env, item.candidate, debug=debug, vis=vis
                )
            except RuntimeError as error:
                last_reason = str(error)
                stage_name = last_reason.split(":", 1)[0]
                stage = {
                    "pregrasp": FailureStage.PREGRASP_PATH,
                    "descend": FailureStage.DESCEND_PATH,
                    "target-penetration": FailureStage.GRIPPER_COLLISION,
                    "target-pad-contact-early": FailureStage.GRIPPER_COLLISION,
                    "grasp": FailureStage.CLOSE,
                    "lift": FailureStage.LIFT,
                }.get(stage_name, FailureStage.GRIPPER_COLLISION)
                evaluations.append(
                    CandidateEvaluation(
                        candidate_id=item.candidate.candidate_id,
                        provider=provider,
                        pipeline=pipeline,
                        rank=execution_rank,
                        failure_stage=stage,
                        failure_reason=last_reason,
                        antipodal_score=item.candidate.proposal_score,
                        clearance_score=item.clearance_score,
                        ik_cost=item.ik_cost,
                        path_length=item.path_length,
                    )
                )
                continue
            info = transition[4] if transition is not None else {}
            final_transition = transition
            robust = transition is not None and _scalar_bool(
                info.get("robust_success_10step", False)
            )
            legacy = transition is not None and _scalar_bool(
                info.get("legacy_success_3step", False)
            )
            evaluations.append(
                CandidateEvaluation(
                    candidate_id=item.candidate.candidate_id,
                    provider=provider,
                    pipeline=pipeline,
                    rank=execution_rank,
                    failure_stage=None if robust else FailureStage.ROBUST_HOLD,
                    failure_reason=None if robust else "robust hold was not reached",
                    antipodal_score=item.candidate.proposal_score,
                    clearance_score=item.clearance_score,
                    ik_cost=item.ik_cost,
                    path_length=item.path_length,
                    max_lift_height=float(
                        np.asarray(info.get("lift_height", 0.0)).reshape(-1)[0]
                    ),
                    legacy_success_3step=legacy,
                    robust_success_10step=robust,
                )
            )
            if robust:
                return ExpertResult(
                    True,
                    "accepted",
                    item.candidate.candidate_id,
                    execution_rank,
                    transition,
                    tuple(evaluations),
                )
            last_reason = "robust_hold"
        return ExpertResult(
            False,
            last_reason,
            None,
            min(len(ranked), MAX_GRASP_CANDIDATES),
            final_transition,
            tuple(evaluations),
        )

    if candidates is None:
        candidates = generate_grasp_candidates(
            base_env.agent,
            base_env.obj,
            np.asarray(base_env.agent.robot.pose.p[0].cpu()),
        )
    candidates = list(candidates)
    if not candidates:
        raise ValueError("At least one grasp candidate is required")
    if len(candidates) > MAX_GRASP_CANDIDATES:
        raise ValueError(f"At most {MAX_GRASP_CANDIDATES} candidates may be tested")

    last_reason = "no_transition"
    for index, candidate in enumerate(candidates, start=1):
        _reset_expert_episode(env, seed)
        try:
            transition = _run_candidate(env, candidate, debug=debug, vis=vis)
        except RuntimeError as error:
            last_reason = str(error)
            continue
        if transition is not None and _scalar_bool(transition[4]["success"]):
            return ExpertResult(
                True, "accepted", candidate.candidate_id, index, transition
            )
        last_reason = (
            "timeout"
            if transition is not None and _scalar_bool(transition[3])
            else "lift"
        )
    return ExpertResult(False, last_reason, None, len(candidates), None)


def solve_provider_cascade(
    env,
    *,
    seed: int,
    providers: Iterable[GraspProviderName | str] = DEFAULT_GRASP_PROVIDER_CASCADE,
    grasp_cache_dir: pathlib.Path | None,
    candidate_pool_limit: int,
    debug: bool = False,
    vis: bool = False,
) -> ExpertResult:
    """Try independent proposal providers through the same strict pipeline."""

    provider_order = tuple(GraspProviderName(provider) for provider in providers)
    if not provider_order:
        raise ValueError("provider cascade must contain at least one provider")
    if len(provider_order) != len(set(provider_order)):
        raise ValueError("provider cascade must not contain duplicate providers")
    if candidate_pool_limit < 1:
        raise ValueError("candidate_pool_limit must be positive")

    attempted_candidates = 0
    evaluations: list[CandidateEvaluation] = []
    failures = []
    for provider in provider_order:
        try:
            result = solve(
                env,
                seed=seed,
                provider=provider,
                pipeline=PipelineName.COMMON,
                grasp_cache_dir=grasp_cache_dir,
                candidate_pool_limit=candidate_pool_limit,
                debug=debug,
                vis=vis,
            )
        except RuntimeError as error:
            reason = str(error)
            if not reason.startswith(EXPECTED_PROVIDER_UNAVAILABLE_PREFIXES):
                raise
            failures.append(f"{provider.value}={reason}")
            continue
        attempted_candidates += result.attempted_candidates
        evaluations.extend(result.evaluations)
        if result.success:
            return dataclasses.replace(
                result,
                attempted_candidates=attempted_candidates,
                evaluations=tuple(evaluations),
            )
        failures.append(f"{provider.value}={result.reason}")

    return ExpertResult(
        success=False,
        reason="provider-cascade-exhausted: " + "; ".join(failures),
        candidate_id=None,
        attempted_candidates=attempted_candidates,
        transition=None,
        evaluations=tuple(evaluations),
    )
