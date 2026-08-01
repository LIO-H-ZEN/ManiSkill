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
  - proxy: only if the machine can't reach HF directly. ``export
    https_proxy=http://... http_proxy=http://...`` (HTTP only, NOT
    ``all_proxy=socks5://``). If `curl -sI https://huggingface.co` works with no
    proxy, run this with no proxy set.

Progress: a global per-object bar shows done / total / remaining / failed.
Listing is done per-category (with retry) so a flaky connection retries a small
call instead of the whole tree. Re-runnable: cached files and ``.complete``
instances are skipped.

Usage:
    python -m mani_skill.examples.download_pick_anything_interndata
    python -m mani_skill.examples.download_pick_anything_interndata --max-workers 16
    python -m mani_skill.examples.download_pick_anything_interndata --dry-run
    python -m mani_skill.examples.download_pick_anything_interndata \\
        --categories omniobject3d-banana google_scan-book
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import tyro
from tqdm import tqdm


@dataclass
class Args:
    categories: list[str] = field(default_factory=list)
    """Restrict to these category dirs; empty = all categories (~106)."""

    max_workers: int = 8
    """Parallel download threads (one object per task)."""

    dry_run: bool = False
    """List what would be downloaded (with a size estimate) without downloading."""

    retries: int = 5
    """Retries per network call (listing + each file download)."""


# Per-instance files the env actually reads. Everything else under an instance
# dir (_grasp_*.npz/.npy, _sim.png, Aligned_obj.usd) is skipped to save ~200 GB.
_KEEP_FILES = ("Aligned.obj", "Aligned.mtl")
_TEXTURES_DIR = "textures"


def _is_keep(path: str) -> bool:
    base = path.split("/")[-1]
    if base in _KEEP_FILES:
        return True
    return f"/{_TEXTURES_DIR}/" in path


def _retry(fn, *, attempts: int, what: str):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # network/SSL/timeout etc.
            last = e
            if i < attempts - 1:
                time.sleep(2 * (i + 1))
    raise RuntimeError(f"{what} failed after {attempts} attempts: {last}") from last


def main(args: Args):
    from huggingface_hub import HfApi, RepoFile, RepoFolder, hf_hub_download

    from mani_skill.envs.tasks.pick_anything.randomization.object_sources import (
        InternDataAssetsSource,
    )

    src = InternDataAssetsSource()
    api = HfApi()
    PREFIX = src.REPO_PREFIX
    CACHE_DIR = src.CACHE_DIR

    only_cats = set(args.categories) if args.categories else None

    # 1. List categories (with retry).
    if only_cats:
        cats = sorted(only_cats)
    else:
        print("Listing categories ...")
        entries = _retry(
            lambda: list(
                api.list_repo_tree(
                    repo_id=src.HF_REPO,
                    repo_type=src.HF_REPO_TYPE,
                    path_in_repo=PREFIX,
                )
            ),
            attempts=args.retries,
            what="list categories",
        )
        cats = sorted(
            e.path.split("/")[-1] for e in entries if isinstance(e, RepoFolder)
        )
    print(f"{len(cats)} categories.")

    # 2. List files per category (with retry), group by instance.
    print("Listing objects ...")
    inst_files: dict[tuple[str, str], list[str]] = {}
    for cat in tqdm(cats, desc="listing", smoothing=0):
        try:
            entries = _retry(
                lambda c=cat: list(
                    api.list_repo_tree(
                        repo_id=src.HF_REPO,
                        repo_type=src.HF_REPO_TYPE,
                        path_in_repo=f"{PREFIX}/{c}",
                        recursive=True,
                    )
                ),
                attempts=args.retries,
                what=f"list {cat}",
            )
        except Exception as e:
            print(f"  WARN: could not list {cat} after retries: {e}")
            continue
        for e in entries:
            if not isinstance(e, RepoFile) or not _is_keep(e.path):
                continue
            parts = e.path[len(PREFIX) + 1 :].split("/")
            if len(parts) < 3:
                continue
            inst_files.setdefault((parts[0], parts[1]), []).append(e.path)

    total = len(inst_files)
    if total == 0:
        print("No objects found. Check proxy / network / license acceptance.")
        return

    # 3. Skip instances already marked complete (e.g. from a previous run).
    todo: list[tuple[str, str, list[str]]] = []
    done = 0
    for (cat, inst), fps in inst_files.items():
        local_root = CACHE_DIR / PREFIX / cat / inst
        if (local_root / "Aligned.obj").exists() and (
            local_root / ".complete"
        ).exists():
            done += 1
        else:
            todo.append((cat, inst, fps))
    print(f"{total} objects total: {done} already cached, {len(todo)} to download.\n")

    if args.dry_run:
        print("Sample (first 15 to download):")
        for cat, inst, _ in todo[:15]:
            print(f"  {cat}/{inst}")
        return

    # 4. Download one object per task: fetch its files (retry each), link
    #    textures, mark complete. Global bar tracks objects done/remaining.
    def do_one(cat: str, inst: str, fps: list[str]) -> None:
        local_root = CACHE_DIR / PREFIX / cat / inst
        for fp in fps:
            _retry(
                lambda p=fp: hf_hub_download(
                    repo_id=src.HF_REPO,
                    repo_type=src.HF_REPO_TYPE,
                    filename=p,
                    local_dir=str(CACHE_DIR),
                ),
                attempts=args.retries,
                what=f"download {cat}/{inst}/{fp.split('/')[-1]}",
            )
        src._link_textures(local_root)
        (local_root / ".complete").touch()

    pbar = tqdm(total=total, initial=done, desc="objects", unit="obj")
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {
            ex.submit(do_one, cat, inst, fps): (cat, inst) for cat, inst, fps in todo
        }
        for fut in as_completed(futs):
            cat, inst = futs[fut]
            try:
                fut.result()
            except Exception as e:
                failures.append((f"{cat}/{inst}", str(e)))
            pbar.update(1)
            pbar.set_postfix(remaining=pbar.total - pbar.n, failed=len(failures))
            pbar.refresh()
    pbar.close()

    # 5. Write manifests from what's actually on disk (so the env enumerates and
    #    samples fully offline -- no HF API calls after this).
    import json

    manifest_dir = CACHE_DIR / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    cats_local = src._scan_local_categories()
    (manifest_dir / "_categories.json").write_text(json.dumps(cats_local))
    for cat in cats_local:
        (manifest_dir / f"{cat}.json").write_text(
            json.dumps(src._scan_local_instances(cat))
        )

    print(
        f"\nDONE: {pbar.n}/{total} objects complete ({len(failures)} failed). "
        f"Manifests written for {len(cats_local)} categories."
    )
    if failures:
        print("\nFailures (first 30):")
        for name, err in failures[:30]:
            print(f"  {name}: {err}")
        print("\nRe-run to retry failed objects (cached files are skipped).")
    else:
        print("\nAll objects cached. PickAnything's interndata source is now offline.")


if __name__ == "__main__":
    main(tyro.cli(Args))
