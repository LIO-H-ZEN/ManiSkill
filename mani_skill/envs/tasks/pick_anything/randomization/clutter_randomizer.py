"""Clutter randomizer for PickAnything.

A separate domain-randomization axis (the fifth, alongside object / table /
floor / lighting) that adds **distractor objects** to the table, mimicking
``PickClutterYCB``. Orthogonal to the other axes: ``--clutter 0`` is the
current behavior, and any object-sources / table / floor / lighting config
composes with any clutter count.

Distractors use an independently configurable source pool, or reuse the target
``object_sources`` pool when no separate pool is supplied. They are built per
env at reconfigure and re-placed every episode at random table positions that
avoid the target. Distractors are **physical** (they collide and affect
dynamics) but are **not** added to the state observation --- mirroring
``PickClutterYCB``, where clutter is a visual/physical obstacle the policy must
handle, not an explicit input. (State obs dimensions are unchanged, so
existing policies still load.)

The count may be fixed (``num_clutter=3``) or random per episode
(``num_clutter=(2, 5)`` -> each env draws N in [2, 5] every episode). Random
counts build the upper bound at reconfigure and hide the unused ones far out of
view each episode, so the count varies per episode without a scene rebuild
(this is what makes random counts useful under ``reconfiguration_freq=0``).
"""

from __future__ import annotations

import re
from typing import Sequence, Tuple, Union

import numpy as np
import torch

from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

import mani_skill.envs.utils.randomization as randomization

from .base import Randomizer
from .object_sources import ObjectSource, resolve_object_source

# where unused (hidden) distractors sit: far corner of the floor, out of camera
# view and away from the workspace. Spread by distractor index so they don't
# perfectly overlap.
_HIDE_X, _HIDE_Y, _HIDE_Z = 40.0, 40.0, -0.9


def parse_clutter_spec(c) -> Union[Tuple[int, int], None]:
    """Parse a clutter spec into a ``(lo, hi)`` tuple, or ``None`` (no clutter).

    - ``int`` N -> ``(N, N)`` (fixed count).
    - ``str`` ``"3"`` -> ``(3, 3)``; ``"random_2_5"`` / ``"random-2-5"`` ->
      ``(2, 5)`` (random per episode); ``"0"`` / ``"off"`` / ``"none"`` -> None.
    - ``None`` -> None.
    """
    if c is None:
        return None
    if isinstance(c, int):
        return (c, c) if c > 0 else None
    if isinstance(c, str):
        s = c.strip().lower()
        if s in ("", "0", "none", "off"):
            return None
        m = re.match(r"^random[_-](\d+)[_-](\d+)$", s)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo <= 0 or lo > hi:
                raise ValueError(
                    f"random clutter range invalid: {c!r} (need 0 < lo <= hi)"
                )
            return (lo, hi)
        n = int(s)  # plain int string
        return (n, n) if n > 0 else None
    raise TypeError(f"clutter spec must be int/str/None, got {c!r}")


class ClutterRandomizer(Randomizer):
    """Add distractor objects per env on the table.

    Args:
        num_clutter: fixed count ``int`` N, or ``(lo, hi)`` for a random count
            per episode (each env draws N in [lo, hi] every episode). The upper
            bound ``hi`` is built at reconfigure; unused ones are hidden.
        sources: sequence of :class:`ObjectSource` instances or string aliases
            (``"cube"``/``"ycb"``/``"interndata"``). The environment may pass
            an independent clutter-only pool or the target-object pool.
        spawn_half_size / spawn_center: xy spawn region (match the object
            randomizer so distractors sit alongside the target).
        min_dist: minimum xy distance between an active distractor and the
            target; violating samples are resampled.
        max_resamples: rejection iterations for the ``min_dist`` constraint.
    """

    def __init__(
        self,
        num_clutter: Union[int, Tuple[int, int]],
        sources: Sequence[Union[ObjectSource, str]],
        spawn_half_size: float = 0.1,
        spawn_center=(0.0, 0.0),
        min_dist: float = 0.06,
        max_resamples: int = 5,
    ):
        if isinstance(num_clutter, int):
            lo = hi = num_clutter
        else:
            lo, hi = num_clutter
        if lo <= 0 or lo > hi:
            raise ValueError(f"num_clutter invalid: {num_clutter!r}")
        self.lo, self.hi = int(lo), int(hi)
        self.sources: list[ObjectSource] = [resolve_object_source(s) for s in sources]
        if len(self.sources) == 0:
            raise ValueError("ClutterRandomizer needs at least one source")
        self.spawn_half_size = spawn_half_size
        self.spawn_center = spawn_center
        self.min_dist = min_dist
        self.max_resamples = max_resamples

    @property
    def num_clutter(self) -> Tuple[int, int]:
        return (self.lo, self.hi)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def on_reconfigure(self, env, options: dict) -> None:
        # build the upper bound per env so a random count can activate a subset
        # each episode without rebuilding the scene.
        b = env.num_envs
        H = self.hi
        rng = env._batched_episode_rng

        distractors: list[Actor] = []
        for i in range(b):
            for j in range(H):
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
        env.clutter_count = (self.lo, self.hi)

    def on_after_reconfigure(self, env, options: dict) -> None:
        # resting height per distractor (same trick as the target object)
        zs = []
        for d in env._clutter_objs:
            mesh = d.get_first_collision_mesh()
            zs.append(-mesh.bounding_box.bounds[0, 2])
        env.clutter_zs = common.to_tensor(zs, device=env.device)

    def on_initialize_episode(self, env, env_idx: torch.Tensor, options: dict) -> None:
        self._relocate_clutter(env, env_idx)

    def on_step(self, env, env_idx: torch.Tensor, options: dict) -> None:
        # mid-episode: re-draw which distractors are active and re-place them
        # on the table (still avoiding the target). The pool is pre-built at
        # reconfigure, so this is a pure pose write --- no scene rebuild.
        self._relocate_clutter(env, env_idx)

    def _relocate_clutter(self, env, env_idx: torch.Tensor) -> None:
        with torch.device(env.device):
            b = len(env_idx)
            H = self.hi

            # per-env active count N_i in [lo, hi]
            if self.lo == self.hi:
                N = torch.full((b,), self.lo, dtype=torch.long, device=env.device)
            else:
                N = torch.randint(self.lo, self.hi + 1, (b,), device=env.device)

            # resting z per distractor for the reset envs (env-major rows e*H..)
            rows = env_idx.unsqueeze(1) * H + torch.arange(
                H, device=env.device
            ).unsqueeze(0)
            z_all = env.clutter_zs[rows.reshape(-1)].reshape(b, H)  # (b, H)

            # target xy for the reset envs -> active distractors avoid it
            target_xy = env.obj.pose.p[env_idx][:, :2]  # (b, 2)

            pos_table = self._sample_avoiding_target(b, H, target_xy)  # (b, H, 2)
            hide_xy = torch.zeros((H, 2), device=env.device)
            hide_xy[:, 0] = _HIDE_X + torch.arange(H, device=env.device) * 0.3
            hide_xy[:, 1] = _HIDE_Y

            active = torch.arange(H, device=env.device).unsqueeze(0) < N.unsqueeze(
                1
            )  # (b, H)
            pos = torch.where(
                active.unsqueeze(-1), pos_table, hide_xy.unsqueeze(0).expand(b, H, 2)
            )
            z = torch.where(active, z_all, torch.full_like(z_all, _HIDE_Z))
            qs = randomization.random_quaternions(
                b * H, lock_x=True, lock_y=True
            ).reshape(b, H, 4)

            xyz = torch.zeros((b, H, 3))
            xyz[..., :2] = pos
            xyz[..., 2] = z
            pq = torch.cat([xyz, qs], dim=-1).reshape(b * H, 7)
            if not env.gpu_sim_enabled:
                for local_index, environment_index in enumerate(env_idx.tolist()):
                    start = local_index * H
                    actor_start = environment_index * H
                    for clutter_index in range(H):
                        env._clutter_objs[actor_start + clutter_index].set_pose(
                            Pose.create(pq[start + clutter_index])
                        )
                return
            previous_reset_mask = env.scene._reset_mask.clone()
            env.scene._reset_mask[:] = False
            env.scene._reset_mask[env_idx] = True
            try:
                env.clutter_objs.set_pose(Pose.create(pq))
            finally:
                env.scene._reset_mask = previous_reset_mask

    # ------------------------------------------------------------------ #
    # placement helpers
    # ------------------------------------------------------------------ #
    def _sample_avoiding_target(
        self, b: int, H: int, target_xy: torch.Tensor
    ) -> torch.Tensor:
        lo, hi = -self.spawn_half_size, self.spawn_half_size
        cx, cy = self.spawn_center

        def fresh() -> torch.Tensor:
            p = torch.rand((b, H, 2)) * (hi - lo) + lo
            p[..., 0] += cx
            p[..., 1] += cy
            return p

        pos = fresh()
        for _ in range(self.max_resamples):
            dist = torch.linalg.norm(pos - target_xy.unsqueeze(1), dim=-1)  # (b, H)
            bad = dist < self.min_dist
            if not bool(bad.any()):
                break
            pos = torch.where(bad.unsqueeze(-1), fresh(), pos)
        return pos
