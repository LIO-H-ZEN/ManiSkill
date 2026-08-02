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

import os
from pathlib import Path
from typing import Optional

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from .base import Randomizer


class WoodTableRandomizer(Randomizer):
    """The fixed wood PickCube table, via :class:`TableSceneBuilder`.

    This is the faithful PickCube surface: ``table.glb`` scaled so the top sits
    at z=0, plus the robot-init-qpos logic that ``TableSceneBuilder.initialize``
    owns. ``on_initialize_episode`` delegates to ``table_scene.initialize`` ---
    which sets the table pose **and** resets the robot --- so the env itself does
    not need separate robot-init code.

    The wood texture is fixed (PBR swap is the procedural randomizer below /
    a future texture step); this randomizer is about giving PickAnything the
    same realistic wood table PickCube has.
    """

    def __init__(self, robot_init_qpos_noise: float = 0.02):
        self.robot_init_qpos_noise = robot_init_qpos_noise

    def on_reconfigure(self, env, options: dict) -> None:
        env.table_scene = TableSceneBuilder(
            env, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        env.table_scene.build()
        env.table = env.table_scene.table

    def on_initialize_episode(self, env, env_idx, options: dict) -> None:
        # sets table pose + robot init qpos/pose (PickCube behavior)
        env.table_scene.initialize(env_idx)


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


# ---------------------------------------------------------------------------- #
# Real surface textures (InternDataAssets) + independent physics randomization
# ---------------------------------------------------------------------------- #
def _read_json_list(path: Path) -> Optional[list[str]]:
    """Return the JSON list in ``path``, or None if missing/corrupt/empty.

    Corrupt files (e.g. a manifest left empty by an interrupted write) are
    deleted so they rebuild next time.
    """
    if not path.exists():
        return None
    import json

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        try:
            path.unlink()
        except OSError:
            pass
        return None
    if not isinstance(data, list) or len(data) == 0:
        return None
    return [str(x) for x in data]


class TableTextureSource:
    """Random surface texture for the table from InternDataAssets.

    **Which folder?** InternDataAssets has several texture dirs, and they are
    very different in content:

    - ``background_textures`` (**default**, ~101, 2--4K JPG) --- real tileable
      surface materials: wood, marble, concrete, brick, carpet, ... These are
      the right choice for a realistic table surface. ``floor_textures`` (~16)
      is a subset of these.
    - ``table_textures`` (~896 JPG) --- despite the name these are **COCO-style
      photos** (people, food, trains), i.e. distractor backdrops, not seamless
      surfaces. Use this only if you want random-photo visual distractors.
    - ``dark_table_textures`` (~5 large PNG) --- dark surfaces.

    The dataset ships **no** physical metadata, so this source is visual-only.
    Files are listed via the HF tree API (cached to a manifest) and downloaded
    on demand, cached under ``ASSET_DIR/intern_data_assets/`` (shared with the
    object cache). Falls back to scanning the local cache if the HF API is
    unreachable (offline machine with pre-downloaded textures).
    """

    HF_REPO = "InternRobotics/InternData-A1"
    HF_REPO_TYPE = "dataset"
    CACHE_ROOT = ASSET_DIR / "intern_data_assets"
    _IMG_EXT = (".jpg", ".jpeg", ".png")

    def __init__(
        self,
        subdir: str = "background_textures",
        textures: Optional[list[str]] = None,
        max_texture_dim: int = 1024,
    ):
        """
        Args:
            subdir: which InternDataAssets texture folder to use. Default
                ``"background_textures"`` (real surface materials). Alternatives:
                ``"floor_textures"``, ``"dark_table_textures"``,
                ``"table_textures"`` (COCO photo distractors).
            textures: explicit list of repo-relative texture filenames to sample
                from. ``None`` = discover all via HF / local scan.
            max_texture_dim: textures larger than this on their long side are
                downscaled (and re-encoded to PNG) before being handed to SAPIEN.
                InternDataAssets ships 4096px JPGs that sporadically trigger a
                MoltenVK decode failure on macOS, flooding the whole scene green;
                a 1024px cap avoids it and is plenty for a table surface (the
                cameras are 128/512px). Set 0 to disable. Cached, so paid once.
        """
        self.subdir = subdir
        self.REPO_PREFIX = f"InternDataAssets/assets/{subdir}"
        self.textures = textures
        self.max_texture_dim = max_texture_dim
        self._files: Optional[list[str]] = None

    def _cache_file(self) -> Path:
        return self.CACHE_ROOT / "manifest" / f"{self.subdir}.json"

    def _scan_local(self) -> list[str]:
        root = self.CACHE_ROOT / self.REPO_PREFIX
        if not root.is_dir():
            return []
        return sorted(
            p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in self._IMG_EXT
        )

    def _list_files(self) -> list[str]:
        if self.textures is not None:
            return list(self.textures)
        if self._files is not None:
            return self._files

        cached = _read_json_list(self._cache_file())
        if cached is not None:
            self._files = cached
            return cached

        # Try HF tree API; fall back to a local scan of downloaded textures.
        try:
            from huggingface_hub import HfApi, RepoFile

            entries = list(
                HfApi().list_repo_tree(
                    repo_id=self.HF_REPO,
                    repo_type=self.HF_REPO_TYPE,
                    path_in_repo=self.REPO_PREFIX,
                )
            )
            files = [e.path for e in entries if isinstance(e, RepoFile)]
        except Exception as e:
            local = self._scan_local()
            if not local:
                raise RuntimeError(
                    f"Could not list InternDataAssets table textures. No valid "
                    f"manifest, HF API failed, and no local textures at "
                    f"{self.CACHE_ROOT / self.REPO_PREFIX}.\nOriginal error: {e}"
                ) from e
            files = [f"{self.REPO_PREFIX}/{n}" for n in local]

        if not files:
            raise RuntimeError("InternDataAssets table_textures returned no files.")
        self._cache_file().parent.mkdir(parents=True, exist_ok=True)
        import json

        self._cache_file().write_text(json.dumps(files))
        self._files = files
        return files

    def _ensure_safe_size(self, path: Path) -> str:
        """Return a GPU-safe path for ``path``, downscaling + re-encoding if needed.

        Some 4096px InternDataAssets JPGs sporadically trigger a MoltenVK decode
        failure on macOS that floods the whole scene green. Re-encoding to a
        PNG capped at ``max_texture_dim`` on the long side avoids it. No-op if
        the cap is disabled or the file is already small / already converted.
        The converted file is cached beside the original.
        """
        if not self.max_texture_dim or self.max_texture_dim <= 0:
            return str(path)
        safe = path.parent / "_safe" / (path.stem + ".png")
        if safe.exists():
            return str(safe)
        try:
            from PIL import Image

            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((self.max_texture_dim, self.max_texture_dim))
                safe.parent.mkdir(parents=True, exist_ok=True)
                im.save(safe, "PNG")
        except Exception:
            # if anything goes wrong, fall back to the original file
            return str(path)
        return str(safe)

    def get_texture(self, rng) -> str:
        """Sample one texture (downloading it if needed) and return its local path.

        The returned path points at a GPU-safe (downscaled, PNG) copy when
        ``max_texture_dim`` is set.
        """
        files = self._list_files()
        chosen = str(rng.choice(np.array(files)))
        local = self.CACHE_ROOT / chosen  # hf_hub_download preserves the repo path
        if not local.exists():
            from huggingface_hub import hf_hub_download

            try:
                hf_hub_download(
                    repo_id=self.HF_REPO,
                    repo_type=self.HF_REPO_TYPE,
                    filename=chosen,
                    local_dir=str(self.CACHE_ROOT),
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download table texture {chosen}. InternDataAssets is "
                    "gated; run `huggingface-cli login` + accept the license, or "
                    "pre-download via the bulk download script.\n"
                    f"Original error: {e}"
                ) from e
        return self._ensure_safe_size(local)


class TextureTableRandomizer(Randomizer):
    """Real legged PickCube table with a random surface texture + random friction.

    Visual: the actual ``table.glb`` (tabletop + legs) is loaded as the visual,
    and each of its per-part materials is overridden **in place** with a sampled
    InternDataAssets surface texture (default ``background_textures`` --- real
    wood/marble/concrete materials) via :class:`TableTextureSource`. This keeps
    the real table silhouette (not a plain box) while randomizing the surface
    appearance per reconfigure. The dataset ships no PBR/physics metadata, so
    this is a diffuse color map only (no normal/roughness maps).

    Physics: a :class:`sapien.physx.PhysxMaterial` with randomized
    ``static_friction`` / ``dynamic_friction`` / ``restitution`` is attached to
    the table's box collision, **independently** of the visual texture ---
    appearance and contact dynamics are decoupled, which is the right DR setup
    for sim2real (a wood-grain table may be slippery or grippy).

    The table is kinematic and shared across parallel envs; one texture + one
    friction set are drawn per reconfigure (env 0's RNG). Robot init is owned by
    the env, not this randomizer.
    """

    # PickCube table geometry --- mirrors
    # mani_skill.utils.scene_builder.table.scene_builder.TableSceneBuilder so the
    # legged table.glb is reused exactly (scale, collision box, pose); only the
    # material is swapped.
    _TABLE_SCALE = 1.75
    _TABLE_H = 0.9196429
    _TABLE_HALF = (2.418 / 2, 1.209 / 2, 0.9196429 / 2)
    _TABLE_OFFSET = (-0.12, 0, -0.9196429)

    def __init__(
        self,
        robot_init_qpos_noise: float = 0.02,
        texture_source: Optional[TableTextureSource] = None,
        static_friction=(0.3, 0.8),
        dynamic_friction=(0.2, 0.6),
        restitution=(0.0, 0.05),
        roughness: float = 0.85,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.texture_source = texture_source or TableTextureSource()
        self.static_friction = static_friction
        self.dynamic_friction = dynamic_friction
        self.restitution = restitution
        self.roughness = roughness

    @staticmethod
    def _table_glb_path() -> str:
        from mani_skill.utils.scene_builder.table import scene_builder as _tsb

        return str(Path(_tsb.__file__).parent / "assets" / "table.glb")

    def _apply_texture(self, table, tex_path: str) -> None:
        """Override each visual part's material with the sampled texture in place.

        ``table.glb`` loads as one ``RenderShapeTriangleMesh`` with multiple
        parts (tabletop, legs), each carrying its own material. ``shape.material``
        raises when parts differ, so we set per-part.
        """
        rb = table._objs[0].find_component_by_type(
            sapien.render.RenderBodyComponent
        )
        if rb is None:
            return
        for shape in rb.render_shapes:
            for part in getattr(shape, "parts", None) or []:
                mat = part.material
                if mat is None:
                    continue
                mat.set_base_color_texture(
                    sapien.render.RenderTexture2D(filename=tex_path)
                )
                mat.set_metallic(0.0)
                mat.set_roughness(self.roughness)

    def on_reconfigure(self, env, options: dict) -> None:
        # env 0's sub-RNG drives the shared table sample (one texture + one
        # friction set per reconfigure). Only consumes env 0's RNG slice.
        rng = env._batched_episode_rng[0]

        tex_path = self.texture_source.get_texture(rng)
        sf = float(rng.uniform(*self.static_friction))
        df = float(rng.uniform(*self.dynamic_friction))
        rest = float(rng.uniform(*self.restitution))

        physx_mat = sapien.pysapien.physx.PhysxMaterial(
            static_friction=sf, dynamic_friction=df, restitution=rest
        )

        ox, oy, oz = self._TABLE_OFFSET
        builder = env.scene.create_actor_builder()
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, self._TABLE_H / 2]),
            half_size=list(self._TABLE_HALF),
            material=physx_mat,
        )
        builder.add_visual_from_file(
            filename=self._table_glb_path(),
            scale=[self._TABLE_SCALE] * 3,
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        )
        builder.initial_pose = sapien.Pose(
            p=[ox, oy, oz], q=euler2quat(0, 0, np.pi / 2)
        )
        env.table = builder.build_kinematic(name="table-workspace")

        # swap the glb's wood material for the sampled surface texture
        self._apply_texture(env.table, tex_path)

        floor_width = 500 if env.scene.parallel_in_single_scene else 100
        env.ground = build_ground(
            env.scene, floor_width=floor_width, altitude=-(self._TABLE_H)
        )

        # exposed for logging / debugging
        env.table_texture = os.path.basename(tex_path)
        env.table_friction = (sf, df, rest)

    def on_initialize_episode(self, env, env_idx, options: dict) -> None:
        pass  # static kinematic table (pose set at build); env owns robot init


# String alias -> table randomizer, used by the env to accept simple config like
# table="wood" / table="texture".
def resolve_table_randomizer(
    obj, robot_init_qpos_noise: float = 0.02
) -> Randomizer:
    if isinstance(obj, Randomizer):
        return obj
    if isinstance(obj, str):
        key = obj.lower()
        if key == "wood":
            return WoodTableRandomizer(robot_init_qpos_noise=robot_init_qpos_noise)
        if key in ("texture", "textures", "interndata"):
            return TextureTableRandomizer(robot_init_qpos_noise=robot_init_qpos_noise)
        if key == "procedural":
            return ProceduralTableRandomizer()
        raise ValueError(
            f"Unknown table randomizer '{obj}'. Valid: wood, texture, procedural."
        )
    raise TypeError(
        f"table randomizer must be a Randomizer or string alias, got {obj!r}"
    )
