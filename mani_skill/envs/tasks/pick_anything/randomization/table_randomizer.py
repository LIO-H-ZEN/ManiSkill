"""Table / surface randomizers for PickAnything.

v1 builds a box table and randomizes its **PBR material** (base color +
metallic + roughness) across a few material "types" (metal / glossy / matte),
instead of the fixed wood ``table.glb`` that ``TableSceneBuilder`` uses. A
metallic table reflects the HDRI/lights and looks clearly different from a matte
one, so the material axis is visible without any texture download.

Note: procedural params give material *type* variety (metal vs glossy vs matte),
not wood-grain / marble-vein realism. For true realistic surfaces, swap in real
PBR textures (``base_color_texture`` + ``normal_texture`` + ``roughness_texture``)
--- SAPIEN's ``RenderMaterial`` supports them; that is the v3 (InternDataAssets)
step.
"""

from __future__ import annotations

import sapien

from mani_skill.utils.building.ground import build_ground

from .base import Randomizer

# PBR parameter ranges per material type. metal = high metallic, low roughness;
# glossy = low metallic, low roughness (shiny plastic/lacquer); matte = low
# metallic, high roughness (wood/stone/laminate feel).
MATERIAL_PRESETS = {
    "metal": dict(metallic=(0.85, 1.0), roughness=(0.05, 0.25), color=(0.45, 0.75)),
    "glossy": dict(metallic=(0.0, 0.15), roughness=(0.1, 0.3), color=(0.2, 0.9)),
    "matte": dict(metallic=(0.0, 0.15), roughness=(0.6, 0.95), color=(0.2, 0.9)),
}


class ProceduralTableRandomizer(Randomizer):
    """A box table with a randomized PBR material (color + metallic + roughness).

    The table top is at z=0 (matching the PickCube convention, so objects spawn
    at z=half_size). Table and ground are static/kinematic and shared across all
    parallel envs; only the material is randomized per reconfigure.
    """

    def __init__(
        self,
        table_half_size=(0.5, 0.5, 0.4),
        material_types: list[str] | None = None,
    ):
        self.table_half_size = table_half_size
        self.material_types = list(material_types) if material_types else list(
            MATERIAL_PRESETS
        )

    def _sample_material(self, env):
        rng = env._batched_episode_rng
        mtype = self.material_types[int(rng.randint(0, len(self.material_types))[0])]
        p = MATERIAL_PRESETS[mtype]
        metallic = float(rng.uniform(*p["metallic"])[0])
        roughness = float(rng.uniform(*p["roughness"])[0])
        lo, hi = p["color"]
        col = rng.uniform(lo, hi, size=(3,))[0]

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([float(col[0]), float(col[1]), float(col[2]), 1.0])
        mat.set_metallic(metallic)
        mat.set_roughness(roughness)
        return mat, mtype

    def on_reconfigure(self, env, options: dict) -> None:
        hx, hy, hz = self.table_half_size
        mat, mtype = self._sample_material(env)
        env.table_material_type = mtype  # exposed for logging / debugging

        builder = env.scene.create_actor_builder()
        builder.add_box_collision(half_size=[hx, hy, hz])
        builder.add_box_visual(half_size=[hx, hy, hz], material=mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, -hz])  # top surface at z=0
        env.table = builder.build_kinematic(name="table")

        env.ground = build_ground(env.scene, floor_width=100, altitude=-(2 * hz))
