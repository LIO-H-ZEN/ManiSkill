"""Lighting randomizers for PickAnything.

v1 uses the HDRI environment maps that ship with ManiSkill (no download):
``set_environment_map`` is a render-time call, so the HDRI can be swapped every
episode in ``on_initialize_episode`` --- this is the single highest-leverage
visual randomization. Ambient + a directional light are set at reconfigure time
(SAPIEN lights are not easily mutated per episode, so they randomize per
reconfigure). v3 will plug in InternDataAssets' 87-image ``envmap_lib``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import mani_skill
from mani_skill.utils.logging_utils import logger

from .base import Randomizer


def _default_hdri_files() -> list[str]:
    """HDRI/EXR files shipped with the ManiSkill source tree."""
    pkg = Path(mani_skill.__file__).resolve().parent
    candidates = [
        pkg / "assets" / "environment_maps" / "default.hdr",
        pkg / "utils" / "scene_builder" / "replicacad" / "autumn_field_puresky_4k.hdr",
        pkg / "examples" / "benchmarking" / "envs" / "maniskill" / "kloofendal_28d_misty_puresky_1k.hdr",
        pkg / "examples" / "benchmarking" / "envs" / "maniskill" / "overcast.exr",
    ]
    return [str(p) for p in candidates if p.exists()]


class HDRILightingRandomizer(Randomizer):
    """Randomized HDRI environment map + ambient/directional light.

    - ``on_reconfigure``: set ambient light + one randomized directional light
      (random direction and intensity).
    - ``on_initialize_episode``: swap the HDRI environment map per env. This is
      cheap (render-time) and is the main per-episode lighting randomization.

    Note: per-env HDRI uses ``scene.sub_scenes[i]`` and so gives per-env lighting
    diversity on CPU / multi-sub-scene setups. On GPU single-scene
    (``parallel_in_single_scene``) all envs share one env map.
    """

    def __init__(self, hdri_files: list[str] | None = None):
        # None -> ship defaults; an explicit [] disables HDRI randomization.
        self.hdri_files = (
            _default_hdri_files() if hdri_files is None else list(hdri_files)
        )
        if self.hdri_files and sys.platform == "darwin":
            # SAPIEN's environment-map path renders incorrectly under MoltenVK
            # (observed on Apple Silicon, sapien 3.0.3): the decoded HDR floods
            # the scene with green IBL. Skip the env map entirely on macOS.
            logger.warning(
                "HDRILightingRandomizer: set_environment_map is broken on macOS "
                "(MoltenVK) and tints the whole scene green; disabling HDRI "
                "randomization. Pass hdri_files=[] to silence this warning."
            )
            self.hdri_files = []

    def on_reconfigure(self, env, options: dict) -> None:
        rng = env._batched_episode_rng
        env.scene.set_ambient_light([0.3, 0.3, 0.3])

        # one light direction + intensity shared across envs (env 0's sample)
        direction = rng.uniform(-1.0, 1.0, size=(3,))[0]
        direction[2] = -abs(float(direction[2])) - 0.5  # force pointing downward
        intensity = float(rng.uniform(0.6, 1.2)[0])
        env.scene.add_directional_light(
            direction.tolist(),
            [intensity, intensity, intensity],
            shadow=True,
            shadow_scale=5,
            shadow_map_size=2048,
        )

    def on_initialize_episode(self, env, env_idx, options: dict) -> None:
        if not self.hdri_files:
            return
        rng = env._batched_episode_rng
        # one index per env -> shape (num_envs,)
        idxs = rng.randint(0, len(self.hdri_files))
        for i in range(env.num_envs):
            env.scene.sub_scenes[i].set_environment_map(
                self.hdri_files[int(idxs[i])]
            )
