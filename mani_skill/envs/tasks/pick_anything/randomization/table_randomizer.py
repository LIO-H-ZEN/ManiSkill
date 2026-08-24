"""Table randomizers for PickAnything.

All table randomizers keep the real legged PickCube ``table.glb`` silhouette
(tabletop + legs, top at z=0); they differ only in the table's **surface
material**. The **ground/floor** is a separate axis --- see
:class:`FloorRandomizer` (and ``floor_randomizer`` on the env).

- :class:`WoodTableRandomizer` --- fixed wood (keeps ``table.glb``'s own wood
  material).
- :class:`ProceduralTableRandomizer` --- randomized PBR params (base color +
  metallic + roughness) across material types (metal / glossy / matte). No
  download; this is the zero-download material-variety option.
- :class:`TextureTableRandomizer` --- a real table-surface texture sampled
  from InternDataAssets (``dark_table_textures`` + ``light_table_textures``)
  + **independently** randomized friction on the collision. Appearance and
  contact dynamics are decoupled.

The procedural/texture randomizers load ``table.glb`` directly and override its
per-part materials in place (the glb loads as one triangle-mesh render shape
with multiple parts, each carrying its own material).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR
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


class _RealTableRandomizer(Randomizer):
    """Base for randomizers that reuse the real legged ``table.glb`` and only
    swap its visual material per reconfigure.

    Subclasses implement :meth:`_apply_visual` to override the glb's per-part
    materials (texture, PBR params, ...). The table is kinematic, top at z=0,
    shared across parallel envs; robot init is owned by the env, not the
    randomizer. The **ground is not built here** --- it is a separate axis
    (:class:`FloorRandomizer`).
    """

    # PickCube table geometry --- mirrors
    # mani_skill.utils.scene_builder.table.scene_builder.TableSceneBuilder so the
    # legged table.glb is reused exactly (scale, collision box, pose); only the
    # material is swapped.
    _TABLE_SCALE = 1.75
    _TABLE_H = 0.9196429
    _TABLE_HALF = (2.418 / 2, 1.209 / 2, 0.9196429 / 2)
    _TABLE_OFFSET = (-0.12, 0, -0.9196429)

    def __init__(self, robot_init_qpos_noise: float = 0.02):
        self.robot_init_qpos_noise = robot_init_qpos_noise

    @staticmethod
    def _table_glb_path() -> str:
        from mani_skill.utils.scene_builder.table import scene_builder as _tsb

        return str(Path(_tsb.__file__).parent / "assets" / "table.glb")

    def _apply_visual(self, table) -> None:
        """Override the glb's per-part materials. Implemented by subclasses."""
        raise NotImplementedError

    def _resample_visual_state(self, env) -> None:
        """Re-sample the per-reconfigure visual state (texture path / PBR params).

        Override in subclasses that hold mutable visual state read by
        :meth:`_apply_visual`. The base no-op covers the fixed-wood table, which
        has nothing to re-sample. ``on_step`` calls this then ``_apply_visual``
        to hot-swap the tabletop appearance mid-episode. Friction is a collision
        property baked at build and is deliberately NOT re-sampled here.
        """

    def on_step(self, env, env_idx, options: dict) -> None:
        # the table is a single kinematic actor shared across all parallel envs,
        # so its material swap is global: re-sample once (whenever any env hits
        # the cadence) and re-apply to the live RenderBodyComponent. No scene
        # rebuild --- the geometry/collision are immutable, only the material
        # maps on the existing render body are mutated in place.
        self._resample_visual_state(env)
        self._apply_visual(env.table)


    @staticmethod
    def _each_part_material(table):
        """Yield each ``RenderMaterial`` on each visual part of the glb table.

        ``table.glb`` loads as one ``RenderShapeTriangleMesh`` with multiple
        parts (tabletop, legs), each carrying its own material. ``shape.material``
        raises when parts differ, so we iterate parts.
        """
        rb = table._objs[0].find_component_by_type(
            sapien.render.RenderBodyComponent
        )
        if rb is None:
            return
        for shape in rb.render_shapes:
            for part in getattr(shape, "parts", None) or []:
                mat = part.material
                if mat is not None:
                    yield mat

    @staticmethod
    def _clear_pbr_textures(mat) -> None:
        """Drop every PBR texture map on ``mat`` so only scalars/colors show.

        The glb tabletop ships wood-grain base_color/normal/roughness/metallic
        maps; clearing ``base_color_texture`` alone still leaves the
        normal/roughness grain relief, so the wood pattern stays visible.
        """
        for setter in (
            mat.set_base_color_texture,
            mat.set_normal_texture,
            mat.set_roughness_texture,
            mat.set_metallic_texture,
            mat.set_emission_texture,
            mat.set_transmission_texture,
        ):
            try:
                setter(None)
            except Exception:
                pass

    def _build_table(self, env, physx_mat=None) -> None:
        """Build the legged ``table.glb`` + box collision, then override visuals.

        ``physx_mat`` (optional) attaches a friction material to the collision;
        ``None`` leaves the scene default (used by the procedural randomizer).
        """
        ox, oy, oz = self._TABLE_OFFSET
        builder = env.scene.create_actor_builder()
        coll_kwargs = dict(
            pose=sapien.Pose(p=[0, 0, self._TABLE_H / 2]),
            half_size=list(self._TABLE_HALF),
        )
        if physx_mat is not None:
            coll_kwargs["material"] = physx_mat
        builder.add_box_collision(**coll_kwargs)
        builder.add_visual_from_file(
            filename=self._table_glb_path(),
            scale=[self._TABLE_SCALE] * 3,
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        )
        builder.initial_pose = sapien.Pose(
            p=[ox, oy, oz], q=euler2quat(0, 0, np.pi / 2)
        )
        env.table = builder.build_kinematic(name="table-workspace")

        self._apply_visual(env.table)
        # the ground is built by the (separate) floor randomizer

    def on_initialize_episode(self, env, env_idx, options: dict) -> None:
        pass  # static kinematic table (pose set at build); env owns robot init


class WoodTableRandomizer(_RealTableRandomizer):
    """The fixed wood PickCube table.

    Reuses ``table.glb`` via :meth:`_build_table` but does **not** override the
    visual, so the glb's own wood material is kept. The ground is built by the
    floor randomizer.
    """

    def _apply_visual(self, table) -> None:
        pass  # keep the glb's original wood material

    def on_reconfigure(self, env, options: dict) -> None:
        self._build_table(env, physx_mat=None)


class ProceduralTableRandomizer(_RealTableRandomizer):
    """Real legged PickCube table with a randomized PBR material.

    Reuses ``table.glb`` (so the table keeps its silhouette) and overrides each
    part's material with sampled PBR parameters across a few material "types"
    (metal / glossy / matte). No texture download --- this is the zero-download
    material-variety option. Friction is left at the scene default; for
    randomized friction use :class:`TextureTableRandomizer`.
    """

    def __init__(
        self,
        robot_init_qpos_noise: float = 0.02,
        material_types: list[str] | None = None,
    ):
        super().__init__(robot_init_qpos_noise=robot_init_qpos_noise)
        self.material_types = list(material_types) if material_types else list(
            MATERIAL_PRESETS
        )
        self._pbr = None  # set per reconfigure, read by _apply_visual

    def _sample_material(self, env):
        rng = env._batched_episode_rng
        mtype = self.material_types[int(rng.randint(0, len(self.material_types))[0])]
        p = MATERIAL_PRESETS[mtype]
        metallic = float(rng.uniform(*p["metallic"])[0])
        roughness = float(rng.uniform(*p["roughness"])[0])
        lo, hi = p["color"]
        col = rng.uniform(lo, hi, size=(3,))[0]
        color = [float(col[0]), float(col[1]), float(col[2]), 1.0]
        return mtype, color, metallic, roughness

    def _apply_visual(self, table) -> None:
        if self._pbr is None:
            return
        _, color, metallic, roughness = self._pbr
        for mat in self._each_part_material(table):
            # drop the glb's wood base_color/normal/roughness/metallic maps so
            # the flat base_color shows with no wood-grain relief
            self._clear_pbr_textures(mat)
            mat.set_base_color(color)
            mat.set_metallic(metallic)
            mat.set_roughness(roughness)

    def on_reconfigure(self, env, options: dict) -> None:
        mtype, color, metallic, roughness = self._sample_material(env)
        self._pbr = (mtype, color, metallic, roughness)
        env.table_material_type = mtype  # exposed for logging / debugging
        self._build_table(env, physx_mat=None)

    def _resample_visual_state(self, env) -> None:
        # re-sample PBR params (no rebuild); on_step will re-apply via
        # _apply_visual to the live table render body.
        mtype, color, metallic, roughness = self._sample_material(env)
        self._pbr = (mtype, color, metallic, roughness)
        env.table_material_type = mtype



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
    """Random table-surface texture from InternDataAssets.

    InternDataAssets has several texture folders, very different in content:

    - ``dark_table_textures`` (~7), ``light_table_textures`` (~5) --- **real
      table-surface textures** (the default pool). Despite the COCO-style
      numeric filenames, these are genuine tabletop surface images.
    - ``background_textures`` (~101), ``floor_textures`` (~16, subset) --- real
      surface materials but a mix of wood/marble/concrete/brick/carpet; some
      read as floor rather than table. Add these for more DR variety.
    - ``table_textures`` (~896) --- despite the name, **COCO-style photos**
      (people/food/objects), not surfaces. Avoid for a table.

    The dataset ships no physical metadata, so this source is visual-only.
    Files are listed via the HF tree API (cached per subdir to a manifest) and
    downloaded on demand, cached under ``ASSET_DIR/intern_data_assets/``
    (shared with the object cache). Falls back to scanning the local cache if
    the HF API is unreachable (offline machine with pre-downloaded textures).
    """

    HF_REPO = "InternRobotics/InternData-A1"
    HF_REPO_TYPE = "dataset"
    CACHE_ROOT = ASSET_DIR / "intern_data_assets"
    _IMG_EXT = (".jpg", ".jpeg", ".png")

    def __init__(
        self,
        subdir: Union[str, Sequence[str]] = (
            "dark_table_textures",
            "light_table_textures",
        ),
        textures: Optional[list[str]] = None,
        max_texture_dim: int = 1024,
    ):
        """
        Args:
            subdir: one InternDataAssets texture folder, or a list of them to
                pool and sample from. Default pools the two real table-surface
                folders (``dark_table_textures`` + ``light_table_textures``).
                Add ``"background_textures"`` for more (floor-ish) variety.
            textures: explicit list of repo-relative texture paths to sample
                from (overrides subdir discovery). ``None`` = discover all.
            max_texture_dim: textures larger than this on their long side are
                downscaled (and re-encoded to PNG) before being handed to SAPIEN.
                InternDataAssets ships 4096px JPGs that sporadically trigger a
                MoltenVK decode failure on macOS, flooding the whole scene green;
                a 1024px cap avoids it and is plenty for a table surface (the
                cameras are 128/512px). Set 0 to disable. Cached, so paid once.
        """
        self.subdirs = [subdir] if isinstance(subdir, str) else list(subdir)
        self.textures = textures
        self.max_texture_dim = max_texture_dim
        self._files: Optional[list[str]] = None

    def _cache_file(self, subdir: str) -> Path:
        return self.CACHE_ROOT / "manifest" / f"{subdir}.json"

    def _scan_local(self, subdir: str) -> list[str]:
        root = self.CACHE_ROOT / "InternDataAssets" / "assets" / subdir
        if not root.is_dir():
            return []
        return sorted(
            p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in self._IMG_EXT
        )

    def _list_subdir(self, subdir: str) -> list[str]:
        """Full repo paths for one subdir (manifest cache, HF/local fallback)."""
        prefix = f"InternDataAssets/assets/{subdir}"
        cached = _read_json_list(self._cache_file(subdir))
        if cached is not None:
            return cached
        try:
            from huggingface_hub import HfApi
            from huggingface_hub.hf_api import RepoFile

            entries = list(
                HfApi().list_repo_tree(
                    repo_id=self.HF_REPO,
                    repo_type=self.HF_REPO_TYPE,
                    path_in_repo=prefix,
                )
            )
            files = [e.path for e in entries if isinstance(e, RepoFile)]
        except Exception as e:
            local = self._scan_local(subdir)
            if not local:
                raise RuntimeError(
                    f"Could not list InternDataAssets textures under {subdir}. No "
                    f"valid manifest, HF API failed, and no local textures at "
                    f"{self.CACHE_ROOT / prefix}.\nOriginal error: {e}"
                ) from e
            files = [f"{prefix}/{n}" for n in local]

        if not files:
            raise RuntimeError(f"InternDataAssets {subdir} returned no files.")
        self._cache_file(subdir).parent.mkdir(parents=True, exist_ok=True)
        import json

        self._cache_file(subdir).write_text(json.dumps(files))
        return files

    def _list_files(self) -> list[str]:
        if self.textures is not None:
            return list(self.textures)
        if self._files is not None:
            return self._files
        files: list[str] = []
        for subdir in self.subdirs:
            files.extend(self._list_subdir(subdir))
        if not files:
            raise RuntimeError(
                f"InternDataAssets table textures empty for subdirs={self.subdirs}."
            )
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


class TextureTableRandomizer(_RealTableRandomizer):
    """Real legged PickCube table with a random surface texture + random friction.

    Visual: the actual ``table.glb`` (tabletop + legs) is loaded as the visual,
    and each of its per-part materials is overridden **in place** with a sampled
    InternDataAssets table-surface texture (default pools
    ``dark_table_textures`` + ``light_table_textures`` --- genuine tabletop
    surfaces) via :class:`TableTextureSource`. This keeps the real table
    silhouette (not a plain box) while randomizing the surface appearance per
    reconfigure. The dataset ships no PBR/physics metadata, so
    this is a diffuse color map only (no normal/roughness maps).

    Physics: a :class:`sapien.physx.PhysxMaterial` with randomized
    ``static_friction`` / ``dynamic_friction`` / ``restitution`` is attached to
    the table's box collision, **independently** of the visual texture ---
    appearance and contact dynamics are decoupled, which is the right DR setup
    for sim2real (a wood-grain table may be slippery or grippy).

    One texture + one friction set are drawn per reconfigure (env 0's RNG).
    """

    def __init__(
        self,
        robot_init_qpos_noise: float = 0.02,
        texture_source: Optional[TableTextureSource] = None,
        static_friction=(0.3, 0.8),
        dynamic_friction=(0.2, 0.6),
        restitution=(0.0, 0.05),
        roughness: float = 0.85,
    ):
        super().__init__(robot_init_qpos_noise=robot_init_qpos_noise)
        self.texture_source = texture_source or TableTextureSource()
        self.static_friction = static_friction
        self.dynamic_friction = dynamic_friction
        self.restitution = restitution
        self.roughness = roughness
        self._tex_path = None  # set per reconfigure, read by _apply_visual

    def _apply_visual(self, table) -> None:
        if self._tex_path is None:
            return
        for mat in self._each_part_material(table):
            # drop the glb's wood maps (base_color/normal/roughness/metallic)
            # so only the sampled surface texture shows, with no wood-grain relief
            self._clear_pbr_textures(mat)
            mat.set_base_color_texture(
                sapien.render.RenderTexture2D(filename=self._tex_path)
            )
            # no PBR maps available; pick a sane non-metallic, moderately rough surface
            mat.set_metallic(0.0)
            mat.set_roughness(self.roughness)

    def on_reconfigure(self, env, options: dict) -> None:
        # env 0's sub-RNG drives the shared table sample (one texture + one
        # friction set per reconfigure). Only consumes env 0's RNG slice.
        rng = env._batched_episode_rng[0]

        tex_path = self.texture_source.get_texture(rng)
        self._tex_path = tex_path
        sf = float(rng.uniform(*self.static_friction))
        df = float(rng.uniform(*self.dynamic_friction))
        rest = float(rng.uniform(*self.restitution))

        physx_mat = sapien.pysapien.physx.PhysxMaterial(
            static_friction=sf, dynamic_friction=df, restitution=rest
        )
        self._build_table(env, physx_mat=physx_mat)

        # exposed for logging / debugging
        env.table_texture = os.path.basename(tex_path)
        env.table_friction = (sf, df, rest)

    def _resample_visual_state(self, env) -> None:
        # re-sample the surface texture only. Friction is a collision property
        # baked at build and is deliberately NOT re-sampled mid-episode.
        rng = env._batched_episode_rng[0]
        tex_path = self.texture_source.get_texture(rng)
        self._tex_path = tex_path
        env.table_texture = os.path.basename(tex_path)



class CompositeTableRandomizer(Randomizer):
    """Pick one table randomizer per reconfigure and delegate to it.

    Lets a config mix table strategies --- e.g. ``["wood", "texture"]`` uses the
    fixed wood table on some reconfigures and a random textured table on others.
    One sub-randomizer is drawn uniformly per reconfigure (env 0's RNG) and
    receives every subsequent hook until the next reconfigure.
    """

    def __init__(self, randomizers: list[Randomizer]):
        if not randomizers:
            raise ValueError("CompositeTableRandomizer needs at least one randomizer.")
        self.randomizers = list(randomizers)
        self._current: Optional[Randomizer] = None

    def on_reconfigure(self, env, options: dict) -> None:
        rng = env._batched_episode_rng[0]
        self._current = self.randomizers[int(rng.randint(0, len(self.randomizers)))]
        env.table_randomizer_choice = type(self._current).__name__  # debug
        # clear per-strategy debug attrs so they reflect the current choice, not
        # whichever randomizer ran on the previous reconfigure (floor_texture is
        # owned by the floor randomizer, not cleared here)
        for a in ("table_texture", "table_friction", "table_material_type"):
            if hasattr(env, a):
                setattr(env, a, None)
        self._current.on_reconfigure(env, options)

    def on_after_reconfigure(self, env, options: dict) -> None:
        if self._current is not None:
            self._current.on_after_reconfigure(env, options)

    def on_initialize_episode(self, env, env_idx, options: dict) -> None:
        if self._current is not None:
            self._current.on_initialize_episode(env, env_idx, options)

    def on_step(self, env, env_idx, options: dict) -> None:
        if self._current is not None:
            self._current.on_step(env, env_idx, options)



def _resolve_single_table(key: str, robot_init_qpos_noise: float) -> Randomizer:
    key = key.lower()
    if key == "wood":
        return WoodTableRandomizer(robot_init_qpos_noise=robot_init_qpos_noise)
    if key in ("texture", "textures", "interndata"):
        return TextureTableRandomizer(robot_init_qpos_noise=robot_init_qpos_noise)
    if key == "procedural":
        return ProceduralTableRandomizer(robot_init_qpos_noise=robot_init_qpos_noise)
    raise ValueError(
        f"Unknown table randomizer '{key}'. Valid: wood, texture, procedural."
    )


# String alias / list of aliases -> table randomizer, used by the env to accept
# simple config like table="wood" / table="texture" / table=["wood","texture"].
def resolve_table_randomizer(
    obj, robot_init_qpos_noise: float = 0.02
) -> Randomizer:
    if isinstance(obj, Randomizer):
        return obj
    if isinstance(obj, str):
        return _resolve_single_table(obj, robot_init_qpos_noise)
    if isinstance(obj, (list, tuple)):
        if len(obj) == 1:
            return resolve_table_randomizer(obj[0], robot_init_qpos_noise)
        return CompositeTableRandomizer(
            [resolve_table_randomizer(o, robot_init_qpos_noise) for o in obj]
        )
    raise TypeError(
        f"table randomizer must be a Randomizer, string alias, or list of them; "
        f"got {obj!r}"
    )


# ---------------------------------------------------------------------------- #
# Floor (ground) randomizer --- a separate axis from the table
# ---------------------------------------------------------------------------- #
class FloorRandomizer(Randomizer):
    """Builds the ground plane, optionally with a sampled floor texture.

    Independent of the table randomizer: the table randomizer builds only the
    table; this builds the ground at ``altitude = -table_height`` (below the
    ``table.glb``). With a ``texture_source``, the ground uses a sampled
    InternDataAssets floor texture (via ``build_ground(texture_file=...)``);
    without one it uses the default checkered grid.

    The ground is static and shared across parallel envs; one texture is drawn
    per reconfigure (env 0's RNG).
    """

    # below the table.glb (table height = 0.9196429); mirrors TableSceneBuilder
    _FLOOR_ALTITUDE = -0.9196429

    def __init__(self, texture_source: Optional[TableTextureSource] = None):
        self.texture_source = texture_source  # None -> checkered grid

    def on_reconfigure(self, env, options: dict) -> None:
        env.floor_texture = None
        kwargs = {}
        if self.texture_source is not None:
            tex = self.texture_source.get_texture(env._batched_episode_rng[0])
            kwargs["texture_file"] = tex
            env.floor_texture = os.path.basename(tex)
        floor_width = 500 if env.scene.parallel_in_single_scene else 100
        env.ground = build_ground(
            env.scene, floor_width=floor_width, altitude=self._FLOOR_ALTITUDE, **kwargs
        )

    def on_initialize_episode(self, env, env_idx, options: dict) -> None:
        pass  # static ground; pose set at build


# String alias -> floor randomizer. "grid" = checkered grid (no download);
# "texture" = InternDataAssets floor textures (floor_textures + background_textures).
def resolve_floor_randomizer(obj) -> Randomizer:
    if isinstance(obj, Randomizer):
        return obj
    if isinstance(obj, str):
        key = obj.lower()
        if key in ("grid", "default", "none"):
            return FloorRandomizer(texture_source=None)
        if key in ("texture", "textures", "interndata"):
            return FloorRandomizer(
                texture_source=TableTextureSource(
                    subdir=["floor_textures", "background_textures"]
                )
            )
        raise ValueError(
            f"Unknown floor randomizer '{obj}'. Valid: grid, texture."
        )
    raise TypeError(
        f"floor randomizer must be a Randomizer or string alias (grid/texture); "
        f"got {obj!r}"
    )
