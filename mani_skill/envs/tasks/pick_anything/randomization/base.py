"""Domain-randomization abstraction for PickAnything.

A ``Randomizer`` owns one axis of scene diversity (the object, the table, the
lighting, ...) and hooks into two distinct points of the env lifecycle. This
split is not arbitrary --- it mirrors what SAPIEN/ManiSkill lets us change
cheaply:

* ``on_reconfigure`` runs during ``_load_scene`` / ``_load_lighting``, i.e. when
  the whole PhysX scene is rebuilt. Anything that is *baked into the scene
  graph* must be randomized here: object geometry, table model / material.
  SAPIEN ``RenderMaterial`` is set at build time and there is no per-episode
  material-swap API, so colors/textures live here too.

* ``on_initialize_episode`` runs every reset in ``_initialize_episode``. This is
  where cheap, render-time / pose changes go: HDRI environment map, light
  parameters, object pose, goal pose. These do not need a PhysX rebuild.

* ``on_after_reconfigure`` runs once per reconfiguration, immediately after
  ``_load_scene`` finishes and the GPU/scene is live. Use it for things that
  need the built actors to exist but must happen before the first episode
  initializes --- e.g. measuring each object's resting height from its collision
  mesh so spawn poses put it flat on the table.

* ``on_step`` runs every N control steps *during* an episode (not on reset),
  driven by ``PickAnythingEnv._after_control_step``. It hot-swaps render-time /
  pose assets mid-trajectory so the policy must stay robust to a changing
  scene (sim2real). Same cheap-API contract as ``on_initialize_episode`` ---
  HDRI swap, light-direction/intensity mutation, table material swap, clutter
  re-placement --- and MUST NOT trigger a PhysX scene rebuild. ``env_idx`` is
  the batch of envs whose ``_elapsed_steps`` hit the N-step cadence this step.

Keeping the hooks separate is what later (v4) lets us turn the
``on_reconfigure`` part into a scene-provided "placement contract" and run pick
in arbitrary scenes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from mani_skill.envs.tasks.pick_anything.pick_anything_env import PickAnythingEnv


class Randomizer:
    """Base class for a domain-randomization axis.

    Subclasses override one or more hooks. ``env`` is the owning
    ``PickAnythingEnv``; randomizers read from it (``env.scene``,
    ``env._batched_episode_rng``, ``env.num_envs``) and write results back as
    attributes on it (e.g. ``env.obj``, ``env.table``) so the env's task logic
    can reference them.
    """

    def on_reconfigure(self, env: "PickAnythingEnv", options: dict) -> None:
        """Called when the env reconfigures (PhysX scene rebuild).

        Use for anything that needs the scene graph rebuilt: object geometry,
        table model / material, fixed lighting. Runs once per reconfiguration,
        not per episode.
        """
        pass

    def on_after_reconfigure(self, env: "PickAnythingEnv", options: dict) -> None:
        """Called once right after ``_load_scene``, with the scene live.

        Use for post-build measurement that the first episode init depends on
        (e.g. object resting heights from collision meshes). The episode RNG is
        seeded here. Runs once per reconfiguration, not per episode.
        """
        pass

    def on_initialize_episode(
        self, env: "PickAnythingEnv", env_idx: torch.Tensor, options: dict
    ) -> None:
        """Called every reset in ``_initialize_episode``.

        Use for cheap render-time / pose changes: HDRI swap, light parameters,
        object pose, goal pose. ``env_idx`` is the batch of envs being reset.
        """
        pass

    def on_step(
        self, env: "PickAnythingEnv", env_idx: torch.Tensor, options: dict
    ) -> None:
        """Called every N control steps *during* an episode (not on reset).

        Hot-swap render-time / pose assets mid-trajectory so the policy must
        stay robust to a changing scene (sim2real): HDRI swap, light
        direction/intensity, table material, clutter re-placement. Same
        cheap-API contract as ``on_initialize_episode`` --- MUST NOT trigger a
        PhysX scene rebuild. ``env_idx`` is the batch of envs whose
        ``_elapsed_steps`` hit the N-step cadence this step.
        """
        pass
