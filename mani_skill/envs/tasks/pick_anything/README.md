# PickAnything

A pick task with **composable domain randomization**. Same pick task as
`PickCube` / `PickSingleYCB` (grasp an object, move it to a goal), but every
diversity axis — object, table, lighting — is delegated to a pluggable
`Randomizer`. v1 is fully procedural (no asset download).

## Why

`PickCube` hardcodes a red cube, a fixed `table.glb`, and two fixed directional
lights — zero visual diversity. This package factors randomization into
swappable `Randomizer` objects so each axis can be randomized (and later swapped
for real assets) independently.

## The abstraction

```python
class Randomizer:
    def on_reconfigure(self, env, options): ...        # PhysX rebuild time
    def on_initialize_episode(self, env, env_idx, options): ...  # every reset
```

The split mirrors what SAPIEN lets us change cheaply:

| Hook | When | Use for |
|------|------|---------|
| `on_reconfigure` | `_load_scene` / `_load_lighting` (PhysX rebuild) | object geometry, table model/material, fixed lights |
| `on_initialize_episode` | `_initialize_episode` (every reset) | HDRI swap, object/goal pose |

Materials are baked at build time (no per-episode material swap in SAPIEN), so
colors/textures live in `on_reconfigure`. HDRI is a render-time call, so it can
swap every episode — the highest-leverage per-episode visual randomization.

## v1 randomizers (procedural, zero download)

- `ProceduralObjectRandomizer` — random box half-size + random color (reconfigure);
  random xy + yaw + goal pose (episode).
- `ProceduralTableRandomizer` — box table with random base-color material
  (replaces the fixed `table.glb`); shared ground.
- `HDRILightingRandomizer` — random HDRI from the 4 maps shipped with ManiSkill
  (per episode) + random ambient/directional light (reconfigure).

## Run

```bash
# GUI: watch object / table / lighting change each episode
python -m mani_skill.examples.demo_random_action -e PickAnything-v1 --render-mode="human"

# different seed -> different episode
python -m mani_skill.examples.demo_random_action -e PickAnything-v1 --render-mode="human" -s 7

# RGB obs (works on compute cards — default raster shader needs no RT cores)
python -m mani_skill.examples.demo_random_action -e PickAnything-v1 -o rgbd --render-mode="human"

# verify randomization + reproducibility (no GUI)
python -m mani_skill.examples.verify_pick_anything
python -m mani_skill.examples.verify_pick_anything --rgb
```

## Customize

Pass your own randomizer (or `None` to disable an axis):

```python
import gymnasium as gym
import mani_skill.envs
from mani_skill.envs.tasks.pick_anything import (
    PickAnythingEnv, HDRILightingRandomizer,
)

env = gym.make(
    "PickAnything-v1", obs_mode="rgb", num_envs=1,
    lighting_randomizer=HDRILightingRandomizer(hdri_files=["/path/to/my.hdr"]),
)
```

## Notes / limitations (v1)

- **Compute cards (A100/H100):** default `minimal`/`default` shader is Vulkan
  rasterization — RGB rendering works without RT cores (unlike Isaac Lab). Only
  the optional `rt` shader needs RT.
- **HDRI per-episode** uses `scene.sub_scenes[i]`, giving per-env lighting on
  CPU / multi-sub-scene. On GPU `parallel_in_single_scene`, all envs share one
  env map.
- **num_envs > 1:** default `reconfiguration_freq=0` (geometry randomized once
  at creation). Set `reconfiguration_freq>=1` to re-randomize geometry, or rely
  on per-episode HDRI/pose which always varies.
- **Robot:** Panda only (v1). Add robots by extending `SUPPORTED_ROBOTS` and the
  init branch in `_initialize_episode`.

## Roadmap

- **v2** — `YCBObjectRandomizer`: reuse ManiSkill's `mani_skill2_ycb` assets.
- **v3** — InternDataAssets integration: `envmap_lib` (87 HDRIs),
  `floor_textures`/`background_textures`, and `basic`/`art` OBJ/GLB meshes.
- **v4** — placement contract on `Randomizer.on_reconfigure` so pick runs in
  arbitrary scenes (RoboCasa kitchen, ReplicaCAD, …) — true "pick anywhere".
