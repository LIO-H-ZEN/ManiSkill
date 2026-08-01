"""One-time bulk download of every InternDataAssets object PickAnything may use.

PickAnything's ``InternDataAssetsSource`` samples from
``InternDataAssets/assets/pick_and_place/pre-train-pick/assets`` (~106 categories,
~3180 textured mesh objects). This script downloads ALL of them up front so the
env never hits the network during training/eval.

Per instance it downloads only the files the env actually reads (``Aligned.obj``,
``Aligned.mtl``, ``textures/``), skipping the large ``_grasp_*.npz/.npy`` (~60MB
each), ``_sim.png``, and the unused ``Aligned_obj.usd``. Files land under
``ASSET_DIR/intern_data_assets/`` (the exact location the env reads), with the
texture symlinks and ``.complete`` markers the env expects --- so after this
runs the interndata source is fully offline.

Requirements:
  - ``huggingface-cli login`` + accept the license at
    https://huggingface.co/datasets/InternRobotics/InternData-A1
  - if behind a proxy: ``export https_proxy=http://... http_proxy=http://...``
    (do NOT use ``all_proxy=socks5://`` --- huggingface_hub/httpx needs the
    ``socksio`` package for SOCKS).

Re-runnable: ``hf_hub_download`` skips already-cached files, and instances
already marked ``.complete`` are skipped. So just re-run if it gets interrupted.

Usage:
    python -m mani_skill.examples.download_pick_anything_interndata
    python -m mani_skill.examples.download_pick_anything_interndata --max-workers 16
    python -m mani_skill.examples.download_pick_anything_interndata --dry-run
    python -m mani_skill.examples.download_pick_anything_interndata \\
        --categories omniobject3d-banana google_scan-book
"""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import tyro
from tqdm import tqdm


@dataclass
class Args:
    categories: list[str] = field(default_factory=list)
    """Restrict to these category dirs; empty = all categories (~106)."""

    max_workers: int = 8
    """Parallel download threads."""

    dry_run: bool = False
    """List what would be downloaded (with a size estimate) without downloading."""


# Per-instance files the env actually reads. Everything else under an instance
# dir (_grasp_*.npz/.npy, _sim.png, Aligned_obj.usd) is skipped to save ~200 GB.
_KEEP_FILES = ("Aligned.obj", "Aligned.mtl")
_TEXTURES_DIR = "textures"


def _is_keep(path: str) -> bool:
    """True for Aligned.obj / Aligned.mtl / anything under textures/."""
    base = path.split("/")[-1]
    if base in _KEEP_FILES:
        return True
    return f"/{_TEXTURES_DIR}/" in path


def main(args: Args):
    from huggingface_hub import HfApi, RepoFile, hf_hub_download

    from mani_skill.envs.tasks.pick_anything.randomization.object_sources import (
        InternDataAssetsSource,
    )

    src = InternDataAssetsSource()
    api = HfApi()
    PREFIX = src.REPO_PREFIX
    CACHE_DIR = src.CACHE_DIR

    only_cats = set(args.categories) if args.categories else None

    # 1. One recursive list of the whole pre-train-pick/assets tree.
    print(f"Listing files under {PREFIX} ...")
    files: list[str] = []  # repo-relative paths to download
    cat_to_instances: dict[str, set[str]] = {}  # category -> {instance, ...}
    total_bytes = 0
    listed = 0
    for e in tqdm(
        api.list_repo_tree(
            repo_id=src.HF_REPO,
            repo_type=src.HF_REPO_TYPE,
            path_in_repo=PREFIX,
            recursive=True,
        ),
        desc="listing",
        smoothing=0,
    ):
        listed += 1
        if not isinstance(e, RepoFile):
            continue
        path = e.path
        # path = PREFIX/<category>/<instance>/(Aligned.obj|Aligned.mtl|textures/..)
        rel = path[len(PREFIX) + 1 :]  # <category>/<instance>/...
        parts = rel.split("/")
        if len(parts) < 3:
            continue
        category, instance = parts[0], parts[1]
        if only_cats is not None and category not in only_cats:
            continue
        if not _is_keep(path):
            continue
        files.append(path)
        total_bytes += int(getattr(e, "size", 0) or 0)
        if parts[-1] == "Aligned.obj":
            cat_to_instances.setdefault(category, set()).add(instance)

    instances = {(c, i) for c, iset in cat_to_instances.items() for i in iset}
    n_cat = len(cat_to_instances)
    print(
        f"\n{len(instances)} objects across {n_cat} categories, "
        f"{len(files)} files, ~{total_bytes / 1e9:.1f} GB to download "
        f"(skipped grasp/sim/usd among {listed} listed entries)."
    )

    if args.dry_run:
        print("\nSample (first 15 objects):")
        for cat, inst in sorted(instances)[:15]:
            print(f"  {cat}/{inst}")
        return

    if not files:
        print("Nothing to download.")
        return

    # 2. Download every file in parallel (hf_hub_download skips cached ones).
    print(f"\nDownloading {len(files)} files with {args.max_workers} workers ...")
    failures: list[tuple[str, str]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {
            ex.submit(
                hf_hub_download,
                repo_id=src.HF_REPO,
                repo_type=src.HF_REPO_TYPE,
                filename=path,
                local_dir=str(CACHE_DIR),
            ): path
            for path in files
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="downloading"):
            path = futs[fut]
            try:
                fut.result()
                done += 1
            except Exception as e:
                failures.append((path, str(e)))
    print(f"downloaded {done}/{len(files)} files, {len(failures)} failed")

    # 3. Link textures into each instance root + mark complete (the env's cache
    #    contract), so the interndata source treats these as ready/offline.
    print("\nLinking textures + marking instances complete ...")
    linked = marked = 0
    for cat, inst in tqdm(sorted(instances), desc="finalizing"):
        local_root = CACHE_DIR / PREFIX / cat / inst
        obj_path = local_root / "Aligned.obj"
        if not obj_path.exists():
            failures.append((f"{cat}/{inst}", "Aligned.obj missing after download"))
            continue
        try:
            src._link_textures(local_root)
            linked += 1
        except Exception as e:
            failures.append((f"{cat}/{inst}", f"link_textures: {e}"))
        (local_root / ".complete").touch()
        marked += 1

    # 4. Write the category/instance manifests the env reads to enumerate
    #    objects, so sampling never hits the network either.
    import json

    manifest_dir = CACHE_DIR / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "_categories.json").write_text(
        json.dumps(sorted(cat_to_instances.keys()))
    )
    for cat, iset in cat_to_instances.items():
        (manifest_dir / f"{cat}.json").write_text(json.dumps(sorted(iset)))
    print(
        f"Wrote manifests: {len(cat_to_instances) + 1} files "
        f"(_categories.json + {len(cat_to_instances)} categories)."
    )

    print(
        f"\nDONE: {marked} instances marked complete, {linked} texture-linked. "
        f"{len(failures)} failures."
    )
    if failures:
        print("\nFailures (first 30):")
        for path, err in failures[:30]:
            print(f"  {path}: {err}")
        print("\nRe-run the script to retry (already-cached files are skipped).")
    else:
        print("\nAll objects cached. PickAnything's interndata source is now offline.")


if __name__ == "__main__":
    main(tyro.cli(Args))
