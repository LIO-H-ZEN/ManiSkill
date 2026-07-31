"""Table / surface randomizers for PickAnything.

v1 builds a simple box table with a random base-color material (instead of the
fixed ``table.glb`` that ``TableSceneBuilder`` uses), so the surface color is
randomized on every reconfigure. v3 will swap in real PBR textures from
InternDataAssets behind the same interface.
"""

from __future__ import annotations

import sapien

from mani_skill.utils.building.ground import build_ground

from .base import Randomizer


class ProceduralTableRandomizer(Randomizer):
    """A box table with a randomized base-color material.

    The table top is at z=0 (matching the PickCube convention, so objects spawn
    at z=half_size). Table and ground are static/kinematic and shared across all
    parallel envs; only the material color is randomized per reconfigure.
    """

    def __init__(self, table_half_size=(0.5, 0.5, 0.4)):
        self.table_half_size = table_half_size

    def on_reconfigure(self, env, options: dict) -> None:
        hx, hy, hz = self.table_half_size

        # shared across envs -> take env 0's sample for the single table actor
        rng = env._batched_episode_rng
        color = rng.uniform(0.1, 0.9, size=(3,))[0].tolist()

        builder = env.scene.create_actor_builder()
        builder.add_box_collision(half_size=[hx, hy, hz])
        builder.add_box_visual(
            half_size=[hx, hy, hz],
            material=sapien.render.RenderMaterial(base_color=color + [1.0]),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, -hz])  # top surface at z=0
        env.table = builder.build_kinematic(name="table")

        env.ground = build_ground(env.scene, floor_width=100, altitude=-(2 * hz))
