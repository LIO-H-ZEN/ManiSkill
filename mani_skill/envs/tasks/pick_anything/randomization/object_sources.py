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

from pathlib import Path
from typing import Optional

import numpy as np
import sapien
import sapien.render

from mani_skill import ASSET_DIR
from mani_skill.utils.structs.actor import Actor


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
        self, env, env_idx: int, rng: np.random.RandomState
    ) -> Actor:  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------------------------------------------------------------- #
# Procedural cube
# ---------------------------------------------------------------------------- #
class CubeSource(ObjectSource):
    """A graspable box with randomized size and color. No assets needed."""

    name = "cube"

    def __init__(self, half_size_range=(0.015, 0.03), color_range=(0.1, 1.0)):
        self.half_size_range = half_size_range
        self.color_range = color_range

    def build_actor(self, env, env_idx: int, rng: np.random.RandomState) -> Actor:
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
        return builder.build(name=f"cube-{env_idx}")


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

    def __init__(self, model_ids: Optional[list[str]] = None):
        from mani_skill.utils.io_utils import load_json

        info_path = ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
        if not info_path.exists():
            raise FileNotFoundError(
                "YCB assets not found at "
                f"{info_path}. Download them first:\n"
                "  python -m mani_skill.utils.download_asset ycb"
            )
        all_ids = [k for k in load_json(info_path).keys() if k not in _YCB_EXCLUDE]
        self.model_ids = (
            np.array(model_ids) if model_ids is not None else np.array(all_ids)
        )

    def build_actor(self, env, env_idx: int, rng: np.random.RandomState) -> Actor:
        from mani_skill.utils.building import actors

        model_id = str(rng.choice(self.model_ids))
        builder = actors.get_actor_builder(env.scene, id=f"ycb:{model_id}")
        builder.initial_pose = sapien.Pose()
        builder.set_scene_idxs([env_idx])
        return builder.build(name=f"ycb-{model_id}-{env_idx}")


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
        categories: Optional[list[str]] = None,
        scale: Optional[float] = None,
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

    def _list_category_instances(self, category: str) -> np.ndarray:
        if category in self._instances:
            return self._instances[category]
        cache = self._cache_path(category)
        if cache.exists():
            import json

            ids = np.array(json.loads(cache.read_text()))
            self._instances[category] = ids
            return ids

        from huggingface_hub import RepoFolder

        try:
            entries = self._list_tree(f"{self.REPO_PREFIX}/{category}")
            ids = np.array(
                [e.path.split("/")[-1] for e in entries if isinstance(e, RepoFolder)]
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not list InternDataAssets instances for category "
                f"'{category}' from {self.HF_REPO}.\n"
                "InternDataAssets is gated. Make sure you have (1) a HuggingFace "
                "token set via `huggingface-cli login` or $HF_TOKEN, and (2) "
                "accepted the dataset license at "
                f"https://huggingface.co/datasets/{self.HF_REPO}.\n"
                f"Original error: {e}"
            ) from e
        if len(ids) == 0:
            raise RuntimeError(
                f"InternDataAssets category '{category}' returned no instances."
            )
        cache.parent.mkdir(parents=True, exist_ok=True)
        import json

        cache.write_text(json.dumps(ids.tolist()))
        self._instances[category] = ids
        return ids

    def _list_categories(self) -> list[str]:
        if self.categories is not None:
            return list(self.categories)
        from huggingface_hub import RepoFolder

        cache = self.CACHE_DIR / "manifest" / "_categories.json"
        if cache.exists():
            import json

            cats = json.loads(cache.read_text())
            if cats:
                return cats
        try:
            entries = self._list_tree(self.REPO_PREFIX)
            cats = [e.path.split("/")[-1] for e in entries if isinstance(e, RepoFolder)]
        except Exception as e:
            raise RuntimeError(
                f"Could not list InternDataAssets categories from {self.HF_REPO}.\n"
                "InternDataAssets is gated. Run `huggingface-cli login`, accept the "
                f"license at https://huggingface.co/datasets/{self.HF_REPO}, "
                f"or pass explicit `categories=[...]`.\nOriginal error: {e}"
            ) from e
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

        from huggingface_hub import RepoFile, hf_hub_download

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
    def build_actor(self, env, env_idx: int, rng: np.random.RandomState) -> Actor:
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
        return builder.build(name=f"interndata-{instance}-{env_idx}")


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
