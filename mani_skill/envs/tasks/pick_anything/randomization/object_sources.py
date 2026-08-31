"""Object sources for PickAnything.

An :class:`ObjectSource` is a *source* of pickable actors (a procedural cube,
the cached YCB dataset, a downloaded mesh dataset, ...). It owns three things:

1. **sampling** --- given a per-env numpy RNG, pick which specific object that
   env gets this reconfiguration.
2. **asset availability** --- make sure the chosen asset is on disk, downloading
   it if the source supports that (cube needs nothing; YCB is pre-cached;
   InternDataAssets downloads meshes on demand).
3. **building** --- construct a dynamic SAPIEN actor in sub-scene ``env_idx``
   from the chosen asset and return it.

The :class:`~mani_skill.envs.tasks.pick_anything.randomization.object_randomizer.CompositeObjectRandomizer`
holds a list of sources and, each reconfiguration, samples one source per
parallel env so a single training run can mix cube / YCB / mesh objects freely.

All sources return actors whose **origin is arbitrary**; the resting height
(bottom-of-object to z=0) is measured generically from the collision mesh in
``CompositeObjectRandomizer.on_after_reconfigure`` (the PickSingleYCB trick),
so a source never needs to know its own geometry to place itself flat.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import sapien
import sapien.render
import trimesh

from mani_skill import ASSET_DIR
from mani_skill.envs.tasks.pick_anything.episode_specs import ObjectSpec
from mani_skill.utils.structs.actor import Actor

ROBODOJO_ASSET_ROOT_ENV = "MANISKILL_ROBODOJO_ASSET_ROOT"
ROBODOJO_COACD_CONVERTER_VERSION = "robodojo_usdz_to_maniskill_v4_material_graph_coacd"
ROBODOJO_COACD_PARAMETERS = {
    "threshold": 0.05,
    "max_convex_hull": 32,
    "seed": 0,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def resolve_robodojo_asset_root(root: str | Path | None = None) -> Path:
    """Resolve the converted RoboDojo asset root without hidden fallbacks."""

    value = root if root is not None else os.environ.get(ROBODOJO_ASSET_ROOT_ENV)
    if value is None:
        raise ValueError(
            f"RoboDojo assets require an explicit root or {ROBODOJO_ASSET_ROOT_ENV}"
        )
    resolved = Path(value).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"RoboDojo converted asset root is missing: {resolved}")
    return resolved


class ObjectSource:
    """Base class for a source of pickable actors.

    Subclasses implement :meth:`build_actor`, which samples one object (using
    the supplied per-env ``rng``), ensures it is on disk, builds it into
    sub-scene ``env_idx`` (``builder.set_scene_idxs([env_idx])``) and returns the
    built dynamic :class:`Actor`. ``rng`` is a single ``np.random.RandomState``
    (one env's slice of ``env._batched_episode_rng``), so draws are deterministic
    per env per seed.
    """

    name: str = "object"

    def build_actor(
        self, env, env_idx: int, rng: np.random.RandomState, name: str | None = None
    ) -> Actor:  # pragma: no cover - interface
        raise NotImplementedError


class RoboDojoConvertedObjectSource(ObjectSource):
    """Build one versioned RoboDojo object from the offline conversion cache."""

    name = "robodojo"
    SUPPORTED_CONVERTER_VERSIONS = frozenset(
        {
            "robodojo_usdz_to_maniskill_v2",
            "robodojo_usdz_to_maniskill_v3_material_graph",
            ROBODOJO_COACD_CONVERTER_VERSION,
        }
    )

    def __init__(
        self,
        *,
        asset_type: str,
        category: str,
        object_id: str,
        asset_root: str | Path | None = None,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        mass: float | None = None,
        friction: float | None = None,
    ):
        if asset_type not in {"Rigid", "Clutter"}:
            raise ValueError(f"Unsupported RoboDojo asset_type: {asset_type!r}")
        if not category or not object_id:
            raise ValueError("RoboDojo category and object_id must not be empty")
        scale_array = np.asarray(scale, dtype=np.float64)
        if scale_array.shape != (3,) or not np.all(np.isfinite(scale_array)):
            raise ValueError(
                f"RoboDojo scale must contain three finite values: {scale}"
            )
        if np.any(scale_array <= 0.0):
            raise ValueError(f"RoboDojo scale must be positive: {scale}")
        if mass is not None and (not np.isfinite(mass) or mass <= 0.0):
            raise ValueError(f"RoboDojo mass must be positive: {mass}")
        if friction is not None and (not np.isfinite(friction) or friction < 0.0):
            raise ValueError(f"RoboDojo friction must be non-negative: {friction}")
        self.asset_type = asset_type
        self.category = category
        self.object_id = object_id
        self.asset_root = resolve_robodojo_asset_root(asset_root)
        self.scale = tuple(float(value) for value in scale_array)
        self.mass = None if mass is None else float(mass)
        self.friction = None if friction is None else float(friction)
        self.asset_key = f"{asset_type}/{category}/{object_id}"
        self.asset_dir = self.asset_root / self.asset_key
        self.visual_path = self.asset_dir / "visual.glb"
        self.conversion_path = self.asset_dir / "conversion.json"
        self._validate_conversion()

    def _validate_conversion(self) -> None:
        manifest_path = self.asset_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"RoboDojo conversion manifest is missing: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_manifest_hash = manifest.pop("manifest_sha256", None)
        actual_manifest_hash = hashlib.sha256(
            _canonical_json_bytes(manifest)
        ).hexdigest()
        if stored_manifest_hash != actual_manifest_hash:
            raise RuntimeError(
                f"RoboDojo conversion manifest hash mismatch: {manifest_path}"
            )
        rows = {row["asset_key"]: row for row in manifest.get("assets", [])}
        if self.asset_key not in rows:
            raise KeyError(
                f"RoboDojo asset is not declared in manifest: {self.asset_key}"
            )
        if not self.conversion_path.is_file():
            raise FileNotFoundError(self.conversion_path)
        conversion = json.loads(self.conversion_path.read_text(encoding="utf-8"))
        if conversion != rows[self.asset_key]:
            raise RuntimeError(
                f"RoboDojo asset manifest row differs from conversion.json: {self.asset_key}"
            )
        converter_version = conversion.get("converter_version")
        if converter_version not in self.SUPPORTED_CONVERTER_VERSIONS:
            raise RuntimeError(
                f"Unsupported RoboDojo converter version for {self.asset_key}: "
                f"{converter_version!r}"
            )
        collision_filename = (
            "collision.ply"
            if converter_version == ROBODOJO_COACD_CONVERTER_VERSION
            else "collision.obj"
        )
        self.collision_path = self.asset_dir / collision_filename
        for filename in ("visual.glb", collision_filename):
            path = self.asset_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            if _sha256_file(path) != conversion[f"{filename}_sha256"]:
                raise RuntimeError(f"RoboDojo converted asset hash mismatch: {path}")
        if converter_version == ROBODOJO_COACD_CONVERTER_VERSION:
            self._validate_coacd_collision(conversion)
        self.conversion = conversion

    def _validate_coacd_collision(self, conversion: dict) -> None:
        decomposition = conversion.get("collision_decomposition")
        if not isinstance(decomposition, dict):
            raise TypeError(
                f"RoboDojo v4 asset has no collision decomposition metadata: {self.asset_key}"
            )
        if decomposition.get("algorithm") != "coacd":
            raise RuntimeError(
                f"RoboDojo v4 asset has unsupported collision decomposition: {self.asset_key}"
            )
        if decomposition.get("parameters") != ROBODOJO_COACD_PARAMETERS:
            raise RuntimeError(
                f"RoboDojo v4 asset has unexpected CoACD parameters: {self.asset_key}"
            )
        if not decomposition.get("package_version"):
            raise RuntimeError(
                f"RoboDojo v4 asset has no CoACD package version: {self.asset_key}"
            )
        collision = trimesh.load(self.collision_path, force="mesh", process=False)
        if not isinstance(collision, trimesh.Trimesh):
            raise TypeError(
                f"RoboDojo v4 collision payload is not one mesh: {self.collision_path}"
            )
        components = tuple(collision.split(only_watertight=False))
        expected_hulls = conversion.get("collision_hull_count")
        if expected_hulls != len(components) or expected_hulls is None:
            raise RuntimeError(
                f"RoboDojo v4 collision hull count mismatch: {self.asset_key}"
            )
        if conversion.get("collision_vertices") != len(collision.vertices):
            raise RuntimeError(
                f"RoboDojo v4 collision vertex count mismatch: {self.asset_key}"
            )
        if conversion.get("collision_triangles") != len(collision.faces):
            raise RuntimeError(
                f"RoboDojo v4 collision triangle count mismatch: {self.asset_key}"
            )
        volumes = np.asarray([abs(float(mesh.volume)) for mesh in components])
        if (
            not components
            or not np.all(np.isfinite(volumes))
            or np.any(volumes <= 0.0)
            or any(not mesh.is_watertight or not mesh.is_volume for mesh in components)
        ):
            raise RuntimeError(
                f"RoboDojo v4 collision contains an invalid convex component: {self.asset_key}"
            )
        expected_volume = conversion.get("collision_volume")
        if expected_volume is None or not np.isclose(
            float(np.sum(volumes)), float(expected_volume), rtol=1e-5, atol=1e-12
        ):
            raise RuntimeError(
                f"RoboDojo v4 collision volume mismatch: {self.asset_key}"
            )

    def load_collision_meshes(self) -> tuple[trimesh.Trimesh, ...]:
        collision = trimesh.load(self.collision_path, force="mesh", process=False)
        if not isinstance(collision, trimesh.Trimesh):
            raise TypeError(f"Expected collision mesh: {self.collision_path}")
        if self.conversion["converter_version"] == ROBODOJO_COACD_CONVERTER_VERSION:
            meshes = tuple(collision.split(only_watertight=False))
        else:
            meshes = (collision,)
        scale = np.asarray(self.scale, dtype=np.float64)
        result = []
        for mesh in meshes:
            scaled = mesh.copy()
            scaled.vertices = np.asarray(scaled.vertices, dtype=np.float64) * scale
            result.append(scaled)
        return tuple(result)

    def _collision_density(self) -> float:
        if self.mass is None:
            return 1000.0
        volume = sum(abs(float(mesh.volume)) for mesh in self.load_collision_meshes())
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError(
                f"RoboDojo collision mesh has invalid volume: {self.collision_path}"
            )
        return self.mass / volume

    def build_actor(
        self,
        env,
        env_idx: int,
        rng: np.random.RandomState,
        name: str | None = None,
    ) -> Actor:
        del rng
        material = None
        if self.friction is not None:
            material = sapien.physx.PhysxMaterial(
                static_friction=self.friction,
                dynamic_friction=self.friction,
                restitution=0.0,
            )
        builder = env.scene.create_actor_builder()
        collision_kwargs = {
            "filename": str(self.collision_path),
            "scale": self.scale,
            "material": material,
            "density": self._collision_density(),
        }
        if self.conversion["converter_version"] == ROBODOJO_COACD_CONVERTER_VERSION:
            builder.add_multiple_convex_collisions_from_file(
                **collision_kwargs,
                decomposition="none",
            )
        else:
            builder.add_convex_collision_from_file(**collision_kwargs)
        builder.add_visual_from_file(filename=str(self.visual_path), scale=self.scale)
        builder.initial_pose = sapien.Pose()
        builder.set_scene_idxs([env_idx])
        return builder.build(
            name=name
            or f"robodojo-{self.asset_type.lower()}-{self.category}-{self.object_id}-{env_idx}"
        )


class FixedObjectSource(ObjectSource):
    """Build exactly one versioned object without consuming sampling RNG."""

    name = "fixed"

    def __init__(self, spec: ObjectSpec):
        self.spec = spec

    def build_actor(
        self,
        env,
        env_idx: int,
        rng: np.random.RandomState,
        name: str | None = None,
    ) -> Actor:
        spec = self.spec
        if spec.source == "cube":
            builder = env.scene.create_actor_builder()
            half_size = float(spec.cube_half_size)
            builder.add_box_collision(half_size=[half_size] * 3)
            builder.add_box_visual(
                half_size=[half_size] * 3,
                material=sapien.render.RenderMaterial(base_color=list(spec.cube_color)),
            )
        elif spec.source == "ycb":
            from mani_skill.utils.building import actors

            source = YCBSource(model_ids=[spec.object_id])
            if spec.object_id not in source.model_ids:
                raise ValueError(f"Unknown YCB object ID: {spec.object_id}")
            builder = actors.get_actor_builder(env.scene, id=f"ycb:{spec.object_id}")
        elif spec.source == "interndata":
            source = InternDataAssetsSource(categories=[str(spec.category)])
            instances = source._list_category_instances(str(spec.category))
            if spec.object_id not in instances:
                raise ValueError(
                    f"Unknown InternData object: {spec.category}/{spec.object_id}"
                )
            obj_path = source._download_instance(str(spec.category), spec.object_id)
            scale = source._resolve_scale(obj_path)
            builder = env.scene.create_actor_builder()
            builder.add_multiple_convex_collisions_from_file(
                filename=obj_path, scale=[scale] * 3
            )
            builder.add_visual_from_file(filename=obj_path, scale=[scale] * 3)
        else:
            source = RoboDojoConvertedObjectSource(
                asset_type="Rigid",
                category=str(spec.category),
                object_id=spec.object_id,
            )
            return source.build_actor(env, env_idx, rng, name=name)

        builder.initial_pose = sapien.Pose()
        builder.set_scene_idxs([env_idx])
        return builder.build(name=name or f"{spec.source}-{spec.object_id}-{env_idx}")


# ---------------------------------------------------------------------------- #
# Procedural cube
# ---------------------------------------------------------------------------- #
class CubeSource(ObjectSource):
    """A graspable box with randomized size and color. No assets needed."""

    name = "cube"

    def __init__(self, half_size_range=(0.015, 0.03), color_range=(0.1, 1.0)):
        self.half_size_range = half_size_range
        self.color_range = color_range

    def build_actor(
        self, env, env_idx: int, rng: np.random.RandomState, name: str | None = None
    ) -> Actor:
        hs = float(rng.uniform(*self.half_size_range))
        col = rng.uniform(*self.color_range, size=(3,))
        builder = env.scene.create_actor_builder()
        builder.add_box_collision(half_size=[hs] * 3)
        builder.add_box_visual(
            half_size=[hs] * 3,
            material=sapien.render.RenderMaterial(
                base_color=[float(col[0]), float(col[1]), float(col[2]), 1.0]
            ),
        )
        builder.initial_pose = sapien.Pose()
        builder.set_scene_idxs([env_idx])
        return builder.build(name=name or f"cube-{env_idx}")


# ---------------------------------------------------------------------------- #
# YCB (pre-cached via ManiSkill asset download)
# ---------------------------------------------------------------------------- #
# A few YCB models that are awkward/non-graspable for a parallel-jaw gripper;
# mirror the exclusion PickSingleYCB uses.
_YCB_EXCLUDE = {"022_windex_bottle", "028_skillet_lid", "029_plate", "059_chain"}


class YCBSource(ObjectSource):
    """A random YCB object from the cached ManiSkill YCB dataset.

    Uses :func:`mani_skill.utils.building.actors.get_actor_builder` with
    ``id="ycb:{model_id}"``. YCB must be downloaded once
    (``python -m mani_skill.utils.download_asset ycb``); it lives under
    ``ASSET_DIR/assets/mani_skill2_ycb``.
    """

    name = "ycb"

    def __init__(self, model_ids: list[str] | None = None):
        from mani_skill.utils.io_utils import load_json

        info_path = ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
        if not info_path.exists():
            raise FileNotFoundError(
                "YCB assets not found at "
                f"{info_path}. Download them first:\n"
                "  python -m mani_skill.utils.download_asset ycb"
            )
        all_ids = [k for k in load_json(info_path) if k not in _YCB_EXCLUDE]
        self.model_ids = (
            np.array(model_ids) if model_ids is not None else np.array(all_ids)
        )

    def build_actor(
        self, env, env_idx: int, rng: np.random.RandomState, name: str | None = None
    ) -> Actor:
        from mani_skill.utils.building import actors

        model_id = str(rng.choice(self.model_ids))
        builder = actors.get_actor_builder(env.scene, id=f"ycb:{model_id}")
        builder.initial_pose = sapien.Pose()
        builder.set_scene_idxs([env_idx])
        return builder.build(name=name or f"ycb-{model_id}-{env_idx}")


# ---------------------------------------------------------------------------- #
# InternDataAssets (gated HF dataset, downloaded on demand)
# ---------------------------------------------------------------------------- #
class InternDataAssetsSource(ObjectSource):
    """A random textured mesh object from InternRobotics/InternData-A1.

    The dataset is a **gated** HuggingFace dataset
    (``InternRobotics/InternData-A1`` -> ``InternDataAssets``). Its file list is
    public but file *content* requires (a) a HuggingFace token and (b) accepting
    the dataset license in the browser. So this source:

    * lazily lists candidate object instances via the HF tree API (cached to a
      local JSON so we don't hit the network every reset);
    * downloads every file for a sampled instance (except the large grasp
      ``.npz``/``.npy`` and sim ``.png``) on first use and caches them under
      ``ASSET_DIR/intern_data_assets/...``;
    * uses the object's **real-world size** (the meshes are in millimeters, so
      a Rubik's cube is ~5.7 cm) --- no rescaling/clamping --- and builds it
      with a convex-decomposed (VHACD) collision mesh.

    If the repo is not accessible (no token / license not accepted),
    :meth:`build_actor` raises a ``RuntimeError`` with the exact steps to fix it.
    """

    name = "interndata"

    HF_REPO = "InternRobotics/InternData-A1"
    HF_REPO_TYPE = "dataset"
    # The "pre-train-pick" split is a large set of individual pickable objects
    # (google_scan-*, omniobject3d-*, ~107 categories, many instances each).
    REPO_PREFIX = "InternDataAssets/assets/pick_and_place/pre-train-pick/assets"
    CACHE_DIR = ASSET_DIR / "intern_data_assets"

    def __init__(
        self,
        categories: list[str] | None = None,
        scale: float | None = None,
        unit: str = "mm",
    ):
        """
        Args:
            categories: restrict sampling to these category dirs (e.g.
                ``["omniobject3d-banana", "google_scan-book"]``). ``None`` means
                sample across all categories discovered via the HF tree API.
            scale: fixed uniform scale; if given, overrides the unit conversion.
                Otherwise objects are built at their **real-world size** (no
                rescaling/clamping).
            unit: native unit of the dataset meshes. The InternDataAssets
                pre-train-pick meshes are in **millimeters** (a Rubik's cube is
                ~57 units = 5.7 cm), so the default ``"mm"`` yields real-world
                sizes. One of ``"mm"``, ``"cm"``, ``"m"``.
        """
        self.categories = categories
        self.scale = scale
        self.unit_scale = {"mm": 0.001, "cm": 0.01, "m": 1.0}[unit]
        # category -> np.array of instance ids; populated lazily.
        self._instances: dict[str, np.ndarray] = {}

    # -- HF tree API helper -------------------------------------------------- #
    def _list_tree(self, path_in_repo: str, recursive: bool = False):
        from huggingface_hub import HfApi

        return HfApi().list_repo_tree(
            repo_id=self.HF_REPO,
            repo_type=self.HF_REPO_TYPE,
            path_in_repo=path_in_repo,
            recursive=recursive,
        )

    # -- instance discovery (cached on disk) -------------------------------- #
    def _cache_path(self, category: str) -> Path:
        return self.CACHE_DIR / "manifest" / f"{category}.json"

    @staticmethod
    def _read_json_list(path: Path) -> list[str] | None:
        """Return the JSON list in ``path``, or None if missing/corrupt/empty.

        Corrupt files (e.g. a manifest left empty by an interrupted write or a
        cross-disk ``mv``) are deleted so they get rebuilt next time.
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

    def _scan_local_categories(self) -> list[str]:
        """Category dirs actually present on disk (offline fallback)."""
        root = self.CACHE_DIR / self.REPO_PREFIX
        if not root.is_dir():
            return []
        return sorted(
            p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    def _scan_local_instances(self, category: str) -> list[str]:
        """Instance dirs actually downloaded for a category (offline fallback)."""
        root = self.CACHE_DIR / self.REPO_PREFIX / category
        if not root.is_dir():
            return []
        return sorted(
            p.name
            for p in root.iterdir()
            if p.is_dir() and (p / "Aligned.obj").exists()
        )

    def _list_category_instances(self, category: str) -> np.ndarray:
        if category in self._instances:
            return self._instances[category]
        cache = self._cache_path(category)

        cached = self._read_json_list(cache)
        if cached is not None:
            ids = np.array(cached)
            self._instances[category] = ids
            return ids

        # No valid cache: try the HF tree API, and fall back to scanning the
        # local cache dir if that fails (e.g. offline machine with
        # pre-downloaded assets). Either way, rebuild the manifest.
        ids: list[str] | None = None
        try:
            from huggingface_hub.hf_api import RepoFolder

            entries = self._list_tree(f"{self.REPO_PREFIX}/{category}")
            ids = [e.path.split("/")[-1] for e in entries if isinstance(e, RepoFolder)]
        except Exception as e:
            local = self._scan_local_instances(category)
            if not local:
                raise RuntimeError(
                    f"Could not list InternDataAssets instances for category "
                    f"'{category}'. No valid manifest cache, HF API failed, and no "
                    f"local objects at {self.CACHE_DIR / self.REPO_PREFIX / category}.\n"
                    "If offline, run the bulk download script first "
                    "(`python -m mani_skill.examples.download_pick_anything_interndata`), "
                    "or pass explicit `categories=[...]`.\n"
                    f"Original HF error: {e}"
                ) from e
            ids = local

        ids_arr = np.array(ids)
        if len(ids_arr) == 0:
            raise RuntimeError(
                f"InternDataAssets category '{category}' returned no instances."
            )
        cache.parent.mkdir(parents=True, exist_ok=True)
        import json

        cache.write_text(json.dumps(ids))
        self._instances[category] = ids_arr
        return ids_arr

    def _list_categories(self) -> list[str]:
        if self.categories is not None:
            return list(self.categories)

        cache = self.CACHE_DIR / "manifest" / "_categories.json"
        cached = self._read_json_list(cache)
        if cached is not None:
            return cached

        cats: list[str] | None = None
        try:
            from huggingface_hub.hf_api import RepoFolder

            entries = self._list_tree(self.REPO_PREFIX)
            cats = [e.path.split("/")[-1] for e in entries if isinstance(e, RepoFolder)]
        except Exception as e:
            local = self._scan_local_categories()
            if not local:
                raise RuntimeError(
                    f"Could not list InternDataAssets categories. No valid manifest "
                    f"cache, HF API failed, and no local objects at "
                    f"{self.CACHE_DIR / self.REPO_PREFIX}.\n"
                    "If offline, run the bulk download script first, or pass explicit "
                    "`categories=[...]`.\n"
                    f"Original HF error: {e}"
                ) from e
            cats = local

        if not cats:
            raise RuntimeError("InternDataAssets returned no categories.")
        cache.parent.mkdir(parents=True, exist_ok=True)
        import json

        cache.write_text(json.dumps(cats))
        return cats

    # Files we never need (large grasp data + sim preview image).
    _SKIP_SUFFIXES = ("_grasp_dense.npz", "_grasp_sparse.npy", "_sim.png")

    def _link_textures(self, local_root: Path) -> None:
        """Symlink every file in ``textures/`` into the instance root.

        The dataset's ``.mtl`` files reference textures by bare filename
        (``map_Kd Scan.jpg``) even though the file actually lives in
        ``textures/Scan.jpg``. A root-level symlink makes both ``Scan.jpg`` and
        ``textures/Scan.jpg`` resolve, so SAPIEN finds the texture regardless of
        which form the mtl uses. Idempotent.
        """
        tex_dir = local_root / "textures"
        if not tex_dir.is_dir():
            return
        for tex in tex_dir.iterdir():
            if not tex.is_file():
                continue
            link = local_root / tex.name
            if link.exists() or link.is_symlink():
                continue
            try:
                link.symlink_to(Path("textures") / tex.name)
            except OSError:
                import shutil

                shutil.copy(tex, link)

    # -- download one instance's mesh files ---------------------------------- #
    def _download_instance(self, category: str, instance: str) -> str:
        # hf_hub_download preserves the full repo-relative path under local_dir,
        # so files land at CACHE_DIR / REPO_PREFIX / category / instance / ...
        # (mirroring the repo). local_root must match that so the obj/mtl and
        # any textures (which may sit at the instance root, in textures/, or
        # elsewhere depending on the object) resolve relative to the obj.
        local_root = self.CACHE_DIR / self.REPO_PREFIX / category / instance
        obj_path = local_root / "Aligned.obj"
        marker = local_root / ".complete"
        # Always (re)apply texture symlinks: the mtl's bare-filename references
        # are broken without them, even for already-cached instances.
        self._link_textures(local_root)
        if obj_path.exists() and marker.exists():
            return str(obj_path)

        from huggingface_hub import hf_hub_download
        from huggingface_hub.hf_api import RepoFile

        prefix = f"{self.REPO_PREFIX}/{category}/{instance}"
        local_root.mkdir(parents=True, exist_ok=True)
        try:
            # List every file in the instance dir recursively and download all
            # except the grasp data / sim preview. Texture layout varies across
            # objects (some at root like Scan.jpg, some in textures/), so we
            # grab everything to be safe.
            entries = self._list_tree(prefix, recursive=True)
            for e in entries:
                if not isinstance(e, RepoFile):
                    continue
                base = e.path.split("/")[-1]
                if base.startswith(".") or base.endswith(self._SKIP_SUFFIXES):
                    continue
                hf_hub_download(
                    repo_id=self.HF_REPO,
                    repo_type=self.HF_REPO_TYPE,
                    filename=e.path,
                    local_dir=str(self.CACHE_DIR),
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download InternDataAssets object {category}/{instance}.\n"
                "InternDataAssets is gated. Run `huggingface-cli login` and accept "
                f"the license at https://huggingface.co/datasets/{self.HF_REPO}.\n"
                f"Original error: {e}"
            ) from e
        self._link_textures(local_root)
        marker.touch()
        return str(obj_path)

    # -- scale -------------------------------------------------------------- #
    def _resolve_scale(self, obj_path: str) -> float:
        # Real-world size only: the meshes are in millimeters, so convert to
        # meters and do NOT rescale/clamp --- objects keep their true dimensions.
        # Pass a fixed ``scale`` to the constructor to grow/shrink uniformly.
        if self.scale is not None:
            return float(self.scale)
        return self.unit_scale

    # -- build --------------------------------------------------------------- #
    def build_actor(
        self, env, env_idx: int, rng: np.random.RandomState, name: str | None = None
    ) -> Actor:
        categories = self._list_categories()
        category = str(rng.choice(np.array(categories)))
        instances = self._list_category_instances(category)
        instance = str(rng.choice(instances))

        obj_path = self._download_instance(category, instance)
        scale = self._resolve_scale(obj_path)

        builder = env.scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(
            filename=obj_path, scale=[scale] * 3
        )
        builder.add_visual_from_file(filename=obj_path, scale=[scale] * 3)
        builder.initial_pose = sapien.Pose()
        builder.set_scene_idxs([env_idx])
        return builder.build(name=name or f"interndata-{instance}-{env_idx}")


# String alias -> source class, used by the env to accept simple config like
# object_sources=["cube", "ycb", "interndata"].
SOURCE_ALIASES: dict[str, type[ObjectSource]] = {
    "cube": CubeSource,
    "ycb": YCBSource,
    "interndata": InternDataAssetsSource,
}


def resolve_object_source(obj) -> ObjectSource:
    """Resolve a string alias / ObjectSource instance into an ObjectSource."""
    if isinstance(obj, ObjectSource):
        return obj
    if isinstance(obj, str):
        key = obj.lower()
        if key not in SOURCE_ALIASES:
            raise ValueError(
                f"Unknown object source '{obj}'. Valid aliases: {list(SOURCE_ALIASES)}"
            )
        return SOURCE_ALIASES[key]()
    raise TypeError(
        f"object source must be an ObjectSource or a string alias, got {obj!r}"
    )
