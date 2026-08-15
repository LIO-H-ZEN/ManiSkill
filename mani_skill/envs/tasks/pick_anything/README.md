# PickAnything

A pick task (grasp an object, move it to a goal) with composable domain
randomization. Four independent axes, each a pluggable `Randomizer`:
**object**, **table**, **floor**, **lighting**.

## Parameters

| Flag | Values | Default | Multi? |
|------|--------|---------|--------|
| `--object-sources` | `cube` `ycb` `interndata` | all three | yes (space-sep) |
| `--table-randomizer` | `wood` `texture` `procedural` | `wood` + `texture` (mix) | yes |
| `--floor-randomizer` | `texture` `grid` | `texture` | no |
| `--clutter` | int N or `random_lo_hi` | `0` (off) | no |
| `--domain-rand-freq` | int N (steps) | `25` (`0`=off) | no |

Multi-value axes pick one value per reconfigure. Lighting randomizes by default
(HDRI + light direction/intensity; HDRI auto-disabled on macOS). All flags also
accept `Randomizer` instances in code. `--clutter N` adds N distractor objects
per env (reusing the `--object-sources` pool, placed on the table avoiding the
target); `--clutter random_2_5` draws N in [2,5] **per episode** (builds the
upper bound once, hides the unused ones far out of view each episode, so it
varies under `reconfiguration_freq=0`). Distractors are physical but **not** in
the state observation.

## Mid-episode randomization (`--domain-rand-freq`)

By default randomization only happens on **reset** (`on_reconfigure` bakes
geometry/material; `on_initialize_episode` swaps HDRI / poses). Setting
`--domain-rand-freq N` (default 25, `0` disables) adds a third hook,
`Randomizer.on_step`, that fires **every N control steps during an episode**
(driven from `PickAnythingEnv._after_control_step`) to hot-swap render-time /
pose assets mid-trajectory — forcing the policy to stay robust to a changing
scene (sim2real):

- **lighting** — HDRI env map + directional-light direction/intensity (the
  light handle is captured at reconfigure so `.color`/`.pose` mutate in place;
  on the single-scene GPU setup the directional light is global).
- **table** — re-sampled surface texture / PBR color on the live render body
  (no rebuild; geometry/friction are immutable). The table is a single shared
  actor, so the swap is global.
- **clutter** — active distractor subset + positions re-sampled (teleport on
  the pre-built pool, still avoiding the target).

The **target object and robot are never changed**. The hook is a per-env
boolean mask `(elapsed_steps+1) % N == 0` (false on the reset step), so under
partial reset each env swaps on its own cadence. No PhysX scene rebuild is
triggered. HDRI and the table-texture GPU re-upload need a non-Mac GPU to
verify (HDRI is auto-disabled on macOS/MoltenVK).

## `--object-sources` → data

| Value | Object | Data source | Download |
|-------|--------|-------------|----------|
| `cube` | procedural box (random size/color) | none | no |
| `ycb` | YCB dataset | ManiSkill cache `mani_skill2_ycb` | `python -m mani_skill.utils.download_asset ycb` |
| `interndata` | real-world meshes (mm→m, real size, **no clamping**) | `InternDataAssets/assets/pick_and_place/pre-train-pick/assets/<category>/<instance>/` (`Aligned.obj` + `.mtl` + `textures/`). ~3180 objects, 106 categories | on demand (gated HF) |

## `--table-randomizer` → data

All three keep the real legged `table.glb` (tabletop + legs, top at z=0); they
differ only in the surface material.

| Value | Table surface | Data source | Download |
|-------|---------------|-------------|----------|
| `wood` | fixed wood (`table.glb`'s own material) | `table.glb` (ships with ManiSkill) | no |
| `texture` | sampled table texture + **randomized friction** (independent of texture) | `InternDataAssets/assets/dark_table_textures/` (~7) + `light_table_textures/` (~5) | on demand (gated HF) |
| `procedural` | random PBR (color/metallic/roughness; metal/glossy/matte) | none | no |

## `--floor-randomizer` → data

| Value | Floor | Data source | Download |
|-------|-------|-------------|----------|
| `texture` | sampled floor texture | `InternDataAssets/assets/floor_textures/` (~16) + `background_textures/` (~101) | on demand (gated HF) |
| `grid` | checkered grid | ManiSkill default | no |

> **Heads up:** `InternDataAssets/assets/table_textures/` (~896 files) is
> **COCO photos**, not surface textures — not used, despite the name. Real
> surfaces are only in `dark/light_table_textures` (table) and
> `floor/background_textures` (floor).

## InternDataAssets access

`interndata` objects and `texture` table/floor textures all come from the gated
HuggingFace dataset **`InternRobotics/InternData-A1`** (folder `InternDataAssets`).
File *lists* are public; downloading *content* needs:

1. `huggingface-cli login`
2. Accept the license at <https://huggingface.co/datasets/InternRobotics/InternData-A1>
3. (if proxied) `export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890`
   — HTTP proxies only, **not** `all_proxy=socks5://` (huggingface_hub/httpx
   needs `socksio` for SOCKS).

Files cache under `ASSET_DIR/intern_data_assets/` (`~/.maniskill/data/` by
default; override with the `MS_ASSET_DIR` env var).

**Bulk-download all objects** (recommended before training; ~3180 objects,
~21.8 GB, skips ~200 GB of grasp data):

```bash
python -m mani_skill.examples.download_pick_anything_interndata --dry-run         # preview
python -m mani_skill.examples.download_pick_anything_interndata --max-workers 16  # download
python -m mani_skill.examples.download_pick_anything_interndata --categories omniobject3d-banana  # subset
```

Re-runnable; cached files are skipped. Table/floor textures are small and
download on demand — no bulk step needed.

## Run

```bash
# GUI, all defaults (cube+ycb+interndata objects, wood table, textured floor)
python -m mani_skill.examples.demo_random_action -e PickAnything-v1 --render-mode="human"

# fully no-download
python -m mani_skill.examples.demo_random_action -e PickAnything-v1 --render-mode="human" \
    --object-sources cube ycb --table-randomizer wood --floor-randomizer grid

# textured table + textured floor (sampled independently)
python -m mani_skill.examples.demo_random_action -e PickAnything-v1 --render-mode="human" \
    --table-randomizer texture --floor-randomizer texture

python -m mani_skill.examples.verify_pick_anything        # reproducibility check, no GUI
python -m mani_skill.examples.verify_pick_anything --rgb
```

## Train / evaluate (PPO)

`ppo.py` accepts the same PickAnything flags as the demo (`--object-sources`,
`--table-randomizer`, `--floor-randomizer`); `None` uses the env default.

```bash
bash examples/baselines/ppo/run_state_rl.sh   # state RL (cube+ycb+interndata)
# fully no-download training:
python examples/baselines/ppo/ppo.py --env_id="PickAnything-v1" --num_envs=2048 \
    --object-sources cube ycb --table-randomizer wood --floor-randomizer grid
# eval videos are tagged with the episode's object source: 1_ycb.mp4, 2_cube.mp4, 3_interndata.mp4
python examples/baselines/ppo/ppo.py --env_id="PickAnything-v1" \
    --evaluate --checkpoint=path/to/model.pt --num_eval_envs=1 --num-eval-steps=1000
```

## Notes

- **macOS (Apple Silicon):** HDRI is disabled (MoltenVK floods the scene green);
  light direction/intensity still randomizes. Textures are capped to 1024px
  (same green-flood bug on large JPGs).
- **`num_envs > 1`:** default `reconfiguration_freq=0` → object/table/floor
  sampled once at creation and fixed per env (pose still varies per episode).
  Set `reconfiguration_freq>=1` to re-randomize geometry over time.
- **interndata object sizes** are real-world (mm); some large/awkward shapes are
  hard for the Panda gripper. Filter via `InternDataAssetsSource(categories=[...])`.
- **Robot:** Panda only (v1).

## Customize

```python
import gymnasium as gym
import mani_skill.envs
from mani_skill.envs.tasks.pick_anything.randomization import (
    TextureTableRandomizer, FloorRandomizer, TableTextureSource,
    InternDataAssetsSource,
)
gym.make("PickAnything-v1",
    object_sources=[InternDataAssetsSource(categories=["omniobject3d-banana"])],
    table_randomizer=TextureTableRandomizer(
        texture_source=TableTextureSource(subdir=["dark_table_textures", "background_textures"]),
    ),
    floor_randomizer=FloorRandomizer(texture_source=None),  # grid floor
)
```
