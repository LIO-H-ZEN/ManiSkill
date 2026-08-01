# PickAnything

A pick task with **composable domain randomization**. Same pick task as
`PickCube` / `PickSingleYCB` (grasp an object, move it to a goal), but every
diversity axis - object, table, lighting - is delegated to a pluggable
`Randomizer`. Objects are sampled from a **candidate set** that mixes a
procedural cube, the cached YCB dataset, and InternDataAssets meshes.

## Why

`PickCube` hardcodes a red cube, a fixed `table.glb`, and two fixed directional
lights - zero visual diversity. This package factors randomization into
swappable `Randomizer` objects so each axis can be randomized (and later swapped
for real assets) independently.

## The abstraction

```python
class Randomizer:
    def on_reconfigure(self, env, options): ...        # PhysX rebuild time
    def on_after_reconfigure(self, env, options): ...  # post-build measurement
    def on_initialize_episode(self, env, env_idx, options): ...  # every reset
```

The split mirrors what SAPIEN lets us change cheaply:

| Hook | When | Use for |
|------|------|---------|
| `on_reconfigure` | `_load_scene` / `_load_lighting` (PhysX rebuild) | object geometry/identity, table model/material, fixed lights |
| `on_after_reconfigure` | after `_load_scene`, scene is live | measure resting heights from collision meshes |
| `on_initialize_episode` | `_initialize_episode` (every reset) | HDRI swap, object/goal pose |

Materials are baked at build time (no per-episode material swap in SAPIEN), so
colors/textures live in `on_reconfigure`. HDRI is a render-time call, so it can
swap every episode - the highest-leverage per-episode visual randomization.

## Object sources (cube / YCB / InternDataAssets)

`CompositeObjectRandomizer` holds a list of `ObjectSource`s and samples one per
parallel env per reconfiguration, so a single run mixes sources freely. Each
source's resting height is measured generically from its collision mesh in
`on_after_reconfigure` (the PickSingleYCB trick), so any source places flat.

| Alias | Source | Needs download? |
|-------|--------|-----------------|
| `cube` | procedural box (random size/color) | no |
| `ycb` | ManiSkill cached YCB dataset (`mani_skill2_ycb`) | once (`python -m mani_skill.utils.download_asset ycb`) |
| `interndata` | InternDataAssets meshes (real-world size, mm) | on demand (gated HF repo, see below) |

**Default** candidate set = `["cube", "ycb", "interndata"]`. Select sources via:

```bash
# CLI (demo / verify)
python -m mani_skill.examples.demo_random_action -e PickAnything-v1 \
    --object-sources cube ycb interndata --render-mode="human"
python -m mani_skill.examples.verify_pick_anything --sources cube ycb

# code
gym.make("PickAnything-v1", object_sources=["cube", "ycb"])
```

For a fully no-download default, use `object_sources=["cube", "ycb"]`.

## InternDataAssets: prerequisites & bulk download

`interndata` pulls textured meshes from the **gated** HuggingFace dataset
`InternRobotics/InternData-A1` (folder `InternDataAssets`, split
`pick_and_place/pre-train-pick/assets` - ~3180 objects across 106 categories).
The file *list* is public, but downloading *content* requires:

1. **`huggingface-cli login`** (a HuggingFace token).
2. **Accept the dataset license** in the browser at
   <https://huggingface.co/datasets/InternRobotics/InternData-A1>.
3. **Proxy** (if behind one): `export https_proxy=http://127.0.0.1:7890
   http_proxy=http://127.0.0.1:7890` — use HTTP proxies only, **not**
   `all_proxy=socks5://` (huggingface_hub/httpx needs the `socksio` package for
   SOCKS).

Without these, sampling `interndata` raises a `RuntimeError` with the exact fix.

By default each `interndata` object is downloaded **on first use** and cached
under `ASSET_DIR/intern_data_assets/` (mesh files only - obj/mtl/textures; the
~60 MB grasp `.npz`/`.npy`, sim `.png`, and unused `.usd` are skipped). To
download **all** objects up front so the env never hits the network (recommended
before large-batch training/eval), run:

```bash
# proxy + auth as above, then:
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890

# preview the list + size estimate (no download)
python -m mani_skill.examples.download_pick_anything_interndata --dry-run

# full download: ~3180 objects, ~21.8 GB (skips ~200 GB of grasp data)
python -m mani_skill.examples.download_pick_anything_interndata --max-workers 16

# or download only specific categories
python -m mani_skill.examples.download_pick_anything_interndata \
    --categories omniobject3d-banana google_scan-book
```

The script caches files at the exact path the env reads, links textures into
each instance root (the dataset `.mtl` references textures by bare filename),
writes `.complete` markers, and writes the category/instance manifests - so after
it finishes the `interndata` source is **fully offline** (no HF API calls).
It is **re-runnable**: interrupted runs can be restarted; already-cached
files/markers are skipped.

## Run

```bash
# GUI: watch object / table / lighting change each episode
python -m mani_skill.examples.demo_random_action -e PickAnything-v1 --render-mode="human"

# different seed -> different episode
python -m mani_skill.examples.demo_random_action -e PickAnything-v1 --render-mode="human" -s 7

# RGB obs (works on compute cards - default raster shader needs no RT cores)
python -m mani_skill.examples.demo_random_action -e PickAnything-v1 -o rgbd --render-mode="human"

# verify randomization + reproducibility (no GUI)
python -m mani_skill.examples.verify_pick_anything
python -m mani_skill.examples.verify_pick_anything --rgb
```

## Train / evaluate (PPO)

```bash
# state RL, cube+ycb only (no InternData download during large-batch training)
bash examples/baselines/ppo/run_state_rl.sh
# (the script passes --object-sources cube ycb; add interndata once bulk-downloaded)

# evaluate a checkpoint; eval videos are tagged with the episode's object source
# e.g. 1_ycb.mp4, 2_cube.mp4, 3_interndata.mp4
python examples/baselines/ppo/ppo.py --env_id="PickAnything-v1" \
    --evaluate --checkpoint=path/to/model.pt --num_eval_envs=1 --num-eval-steps=1000
```

## Customize

Pass your own randomizer (or `None` to disable an axis), or a custom
`ObjectSource`:

```python
import gymnasium as gym
import mani_skill.envs
from mani_skill.envs.tasks.pick_anything import (
    PickAnythingEnv, HDRILightingRandomizer, InternDataAssetsSource,
)

env = gym.make(
    "PickAnything-v1", obs_mode="rgb", num_envs=1,
    lighting_randomizer=HDRILightingRandomizer(hdri_files=["/path/to/my.hdr"]),
    # restrict interndata to a few categories:
    object_sources=[InternDataAssetsSource(categories=["omniobject3d-banana"])],
)
```

## Notes / limitations

- **macOS (Apple Silicon):** `set_environment_map` is broken under MoltenVK
  (sapien 3.0.3) - it floods the scene with green IBL. `HDRILightingRandomizer`
  auto-disables HDRI on darwin; light-direction/intensity randomization still
  runs. Real HDRI/RTX validation needs a non-Mac GPU.
- **Compute cards (A100/H100):** default `minimal`/`default` shader is Vulkan
  rasterization - RGB rendering works without RT cores (unlike Isaac Lab). Only
  the optional `rt` shader needs RT.
- **InternData object sizes** are real-world (meshes are in mm). Objects range
  ~2-19 cm; some large/awkward shapes may be hard for the Panda gripper to
  grasp, so success on those will be low. Filter via
  `InternDataAssetsSource(categories=[...])` if needed.
- **num_envs > 1:** default `reconfiguration_freq=0` (geometry randomized once
  at creation). Set `reconfiguration_freq>=1` to re-randomize geometry, or rely
  on per-episode HDRI/pose which always varies. With `interndata` in the source
  list and many envs, the first reconfigure downloads one mesh per env - run the
  bulk download first or use `object_sources=["cube","ycb"]`.
- **Robot:** Panda only (v1). Add robots by extending `SUPPORTED_ROBOTS` and the
  init branch in `_initialize_episode`.

## Roadmap

- **Object realism:** InternDataAssets textures for the **table** and **lighting**
  (`envmap_lib` HDRIs, `table_textures`/`floor_textures`) - not just objects.
- **Graspability filter:** skip InternData objects too large/awkward for the
  gripper so success is meaningful.
- **Placement contract** on `Randomizer.on_reconfigure` so pick runs in arbitrary
  scenes (RoboCasa kitchen, ReplicaCAD, …) - true "pick anywhere".
