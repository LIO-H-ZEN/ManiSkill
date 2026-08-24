"""Lighting randomizers for PickAnything.

v1 uses the HDRI environment maps that ship with ManiSkill (no download):
``set_environment_map`` is a render-time call, so the HDRI can be swapped every
episode in ``on_initialize_episode`` --- this is the single highest-leverage
visual randomization. Ambient + a directional light are set at reconfigure time
and, unlike before, the directional-light component handle is retained so it can
also be re-randomized mid-episode in ``on_step``. v3 will plug in
InternDataAssets' 87-image ``envmap_lib``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sapien

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


def _sample_directional_light(rng):
    """Sample a downward direction + intensity for the directional light.

    Returns ``(direction, intensity)`` mirroring the original on_reconfigure
    sampling so on_step re-randomizes with the same distribution.
    """
    direction = rng.uniform(-1.0, 1.0, size=(3,))[0]
    direction[2] = -abs(float(direction[2])) - 0.5  # force pointing downward
    intensity = float(rng.uniform(0.6, 1.2)[0])
    return direction, intensity


class HDRILightingRandomizer(Randomizer):
    """Randomized HDRI environment map + ambient/directional light.

    - ``on_reconfigure``: set ambient light + one randomized directional light
      (random direction and intensity). The directional-light component handle
      is retained (``self._dir_lights``) so it can be mutated mid-episode.
    - ``on_initialize_episode``: swap the HDRI environment map per env. This is
      cheap (render-time) and is the main per-episode lighting randomization.
    - ``on_step``: re-swap the HDRI and re-randomize the directional light
      direction/intensity for the envs whose step count hit the cadence.

    Note: per-env HDRI uses ``scene.sub_scenes[i]`` and so gives per-env lighting
    diversity on CPU / multi-sub-scene setups. On GPU single-scene
    (``parallel_in_single_scene``) all envs share one env map and one
    directional light, so lighting-direction randomization is global there.
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
        # directional-light component handles, one per sub_scene that owns one.
        # Built in on_reconfigure so on_step can mutate .color/.pose directly
        # (the scene.add_directional_light wrapper returns None and discards it).
        self._dir_lights: list = []

    def on_reconfigure(self, env, options: dict) -> None:
        rng = env._batched_episode_rng
        env.scene.set_ambient_light([0.3, 0.3, 0.3])

        direction, intensity = _sample_directional_light(rng)
        env.directional_light_direction = np.asarray(direction, dtype=np.float64)
        env.directional_light_intensity = float(intensity)
        env.hdri_files = [None] * env.num_envs
        self._dir_lights = self._build_directional_light(env, direction, intensity)

    def _build_directional_light(self, env, direction, intensity):
        """Build one directional-light entity per sub_scene and keep the handles.

        Replicates ``env.scene.add_directional_light`` (scene.py:629-674) but
        returns the ``RenderDirectionalLightComponent`` for each scene so
        ``on_step`` can mutate ``.color`` / ``.pose`` without a rebuild.
        """
        color = [intensity, intensity, intensity]
        lights: list = []
        scene_idxs = list(range(len(env.scene.sub_scenes)))
        for scene_idx in scene_idxs:
            if env.scene.parallel_in_single_scene:
                sub = env.scene.sub_scenes[0]
            else:
                sub = env.scene.sub_scenes[scene_idx]
            entity = sapien.Entity()
            entity.name = "directional_light"
            light = sapien.render.RenderDirectionalLightComponent()
            entity.add_component(light)
            light.color = color
            light.shadow = True
            light.shadow_near = -10.0
            light.shadow_far = 10.0
            light.shadow_half_size = 5.0
            light.shadow_map_size = 2048
            light.pose = sapien.Pose(
                [0, 0, 0],
                sapien.math.shortest_rotation([1, 0, 0], direction.tolist()),
            )
            sub.add_entity(entity)
            lights.append(light)
            if env.scene.parallel_in_single_scene:
                # one shared light for the single-scene GPU setup, matching
                # scene.add_directional_light's break on parallel_in_single_scene
                break
        return lights

    def _swap_hdri(self, env, env_idx) -> None:
        """Re-sample and set the HDRI environment map for the given envs."""
        if not self.hdri_files:
            return
        rng = env._batched_episode_rng
        for i in env_idx.tolist():
            path = self.hdri_files[int(rng[i].randint(0, len(self.hdri_files)))]
            env.scene.sub_scenes[i].set_environment_map(path)
            env.hdri_files[i] = path

    def on_initialize_episode(self, env, env_idx, options: dict) -> None:
        self._swap_hdri(env, env_idx)

    def on_step(self, env, env_idx, options: dict) -> None:
        # re-swap HDRI per env (cheap, render-time)
        self._swap_hdri(env, env_idx)
        # re-randomize the directional light direction + intensity. On the
        # single-scene GPU setup this is global (one shared light); on CPU /
        # multi-sub-scene it is per sub_scene.
        if len(self._dir_lights) == 0:
            return
        rng = env._batched_episode_rng
        for light in self._dir_lights:
            direction, intensity = _sample_directional_light(rng)
            env.directional_light_direction = np.asarray(direction, dtype=np.float64)
            env.directional_light_intensity = float(intensity)
            light.color = [intensity, intensity, intensity]
            light.pose = sapien.Pose(
                [0, 0, 0],
                sapien.math.shortest_rotation([1, 0, 0], direction.tolist()),
            )
