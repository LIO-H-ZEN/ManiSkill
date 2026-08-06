"""Clutter randomizer for PickAnything.

A separate domain-randomization axis (the fifth, alongside object / table /
floor / lighting) that adds **distractor objects** to the table, mimicking
``PickClutterYCB``. It is orthogonal to the other axes: ``--clutter 0`` is the
current behavior, and any object-sources / table / floor / lighting config
composes with any clutter count.

Distractors reuse the env's ``object_sources`` pool (the same candidate set as
the target). They are built per env at reconfigure (exactly like the target
object) and re-placed every episode at random table positions that avoid the
target. Distractors are **physical** (they collide and affect dynamics) but are
**not** added to the state observation --- mirroring ``PickClutterYCB``, where
clutter is a visual/physical obstacle the policy must handle, not an explicit
input. (State obs dimensions are unchanged, so existing policies still load.)
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch

from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

import mani_skill.envs.utils.randomization as randomization

from .base import Randomizer
from .object_sources import ObjectSource, resolve_object_source


class ClutterRandomizer(Randomizer):
    """Add ``num_clutter`` distractor objects per env on the table.

    Args:
        num_clutter: distractors per parallel env (>= 1).
        sources: sequence of :class:`ObjectSource` instances or string aliases
            (``"cube"``/``"ycb"``/``"interndata"``). Reused from the env's
            object sources so distractors come from the same pool as the target.
        spawn_half_size / spawn_center: xy spawn region (match the object
            randomizer so distractors sit alongside the target).
        min_dist: minimum xy distance between a distractor and the target;
            violating samples are resampled.
        max_resamples: rejection iterations for the ``min_dist`` constraint.
    """

    def __init__(
        self,
        num_clutter: int,
        sources: Sequence[Union[ObjectSource, str]],
        spawn_half_size: float = 0.1,
        spawn_center=(0.0, 0.0),
        min_dist: float = 0.06,
        max_resamples: int = 5,
    ):
        if num_clutter <= 0:
            raise ValueError("num_clutter must be > 0")
        self.num_clutter = int(num_clutter)
        self.sources: list[ObjectSource] = [resolve_object_source(s) for s in sources]
        if len(self.sources) == 0:
            raise ValueError("ClutterRandomizer needs at least one source")
        self.spawn_half_size = spawn_half_size
        self.spawn_center = spawn_center
        self.min_dist = min_dist
        self.max_resamples = max_resamples

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def on_reconfigure(self, env, options: dict) -> None:
        b = env.num_envs
        N = self.num_clutter
        rng = env._batched_episode_rng

        # build N distractors per env (env-major order: env0_d0..env0_dN-1,
        # env1_d0, ...), sampling a source per distractor from the shared pool.
        distractors: list[Actor] = []
        for i in range(b):
            for j in range(N):
                src = self.sources[int(rng[i].randint(0, len(self.sources)))]
                d = src.build_actor(env, i, rng[i], name=f"clutter-{i}-{j}")
                env.remove_from_state_dict_registry(d)
                distractors.append(d)

        env._clutter_objs = distractors
        env.clutter_objs = (
            Actor.merge(distractors, name="clutter")
            if len(distractors) > 1
            else distractors[0]
        )
        # NOTE: distractors are NOT registered in the state-dict registry. They
        # are physical obstacles only (not in the state observation), mirroring
        # PickClutterYCB; a merged actor spanning >1 sub-actor per env would
        # otherwise break get_state (per-scene velocity size != pose size).
        env.clutter_count = N

    def on_after_reconfigure(self, env, options: dict) -> None:
        # resting height per distractor (same trick as the target object)
        zs = []
        for d in env._clutter_objs:
            mesh = d.get_first_collision_mesh()
            zs.append(-mesh.bounding_box.bounds[0, 2])
        env.clutter_zs = common.to_tensor(zs, device=env.device)

    def on_initialize_episode(self, env, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(env.device):
            b = len(env_idx)
            N = self.num_clutter

            # z per distractor for the reset envs (env-major rows e*N..e*N+N-1)
            rows = env_idx.unsqueeze(1) * N + torch.arange(N, device=env.device).unsqueeze(0)
            z = env.clutter_zs[rows.reshape(-1)].reshape(b, N)  # (b, N)

            # target xy for the reset envs -> avoid placing on top of it
            target_xy = env.obj.pose.p[env_idx][:, :2]  # (b, 2)

            pos = self._sample_avoiding_target(b, N, target_xy)  # (b, N, 2)
            qs = randomization.random_quaternions(b * N, lock_x=True, lock_y=True).reshape(b, N, 4)

            xyz = torch.zeros((b, N, 3))
            xyz[..., :2] = pos
            xyz[..., 2] = z
            pq = torch.cat([xyz, qs], dim=-1).reshape(b * N, 7)
            env.clutter_objs.set_pose(Pose.create(pq))

    # ------------------------------------------------------------------ #
    # placement helpers
    # ------------------------------------------------------------------ #
    def _sample_avoiding_target(self, b: int, N: int, target_xy: torch.Tensor) -> torch.Tensor:
        lo, hi = -self.spawn_half_size, self.spawn_half_size
        cx, cy = self.spawn_center

        def fresh() -> torch.Tensor:
            p = torch.rand((b, N, 2)) * (hi - lo) + lo
            p[..., 0] += cx
            p[..., 1] += cy
            return p

        pos = fresh()
        for _ in range(self.max_resamples):
            dist = torch.linalg.norm(pos - target_xy.unsqueeze(1), dim=-1)  # (b, N)
            bad = dist < self.min_dist
            if not bool(bad.any()):
                break
            pos = torch.where(bad.unsqueeze(-1), fresh(), pos)
        return pos
