"""Render the PickAnything table under each procedural material type.

Saves one PNG per material (metal / glossy / matte) so you can eyeball whether
the procedural materials are good enough or whether we need real textures.

Run from the repo root:

    python -m mani_skill.examples.render_table_materials
    python -m mani_skill.examples.render_table_materials --out-dir /tmp/pa_mat --seed 7
"""

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import tyro

from mani_skill.envs.tasks.pick_anything import ProceduralTableRandomizer


@dataclass
class Args:
    out_dir: str = "/tmp/pick_anything_materials"
    seed: int = 7


def render_material(mtype: str, out_path: Path, seed: int):
    import mani_skill.envs  # noqa: register envs

    env = gym.make(
        "PickAnything-v1",
        obs_mode="rgb",
        num_envs=1,
        render_mode="rgb_array",
        table_randomizer=ProceduralTableRandomizer(material_types=[mtype]),
    )
    env.reset(seed=seed)
    frame = env.render()
    img = np.asarray(frame)
    if img.ndim == 4:  # (1, H, W, 3) -> (H, W, 3)
        img = img[0]
    env.close()

    import imageio.v3 as iio

    iio.imwrite(str(out_path), img)
    return img.shape


def main(args: Args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mtype in ["metal", "glossy", "matte"]:
        out = out_dir / f"table_{mtype}.png"
        shape = render_material(mtype, out, args.seed)
        print(f"  {mtype:7s} -> {out}  (shape={shape})")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
