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

Keeping the two hooks separate is what later (v4) lets us turn the
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

    Subclasses override one or both hooks. ``env`` is the owning
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

    def on_initialize_episode(
        self, env: "PickAnythingEnv", env_idx: torch.Tensor, options: dict
    ) -> None:
        """Called every reset in ``_initialize_episode``.

        Use for cheap render-time / pose changes: HDRI swap, light parameters,
        object pose, goal pose. ``env_idx`` is the batch of envs being reset.
        """
        pass
