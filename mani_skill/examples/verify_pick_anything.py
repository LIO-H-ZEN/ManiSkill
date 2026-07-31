"""Verify that PickAnything randomizes across episodes and is reproducible.

Run from the repo root:

    python -m mani_skill.examples.verify_pick_anything
    python -m mani_skill.examples.verify_pick_anything --rgb            # render a frame
    python -m mani_skill.examples.verify_pick_anything --sources ycb
    python -m mani_skill.examples.verify_pick_anything --sources cube ycb

Default sources are cube + YCB (YCB must be downloaded once:
``python -m mani_skill.utils.download_asset ycb``). Add ``interndata`` to
exercise the InternDataAssets mesh source (gated HF repo; needs
``huggingface-cli login`` + license acceptance).
"""

import gymnasium as gym
import numpy as np
import tyro
from dataclasses import dataclass, field


@dataclass
class Args:
    rgb: bool = False
    """Also render one RGB camera frame per episode to confirm rendering works."""
    num_episodes: int = 6
    sources: list[str] = field(default_factory=lambda: ["cube", "ycb"])
    """Object source aliases to mix (cube / ycb / interndata)."""
    seed: int = 123


def main(args: Args):
    import mani_skill.envs  # noqa: register envs

    obs_mode = "rgb" if args.rgb else "state"
    env = gym.make(
        "PickAnything-v1",
        obs_mode=obs_mode,
        num_envs=1,
        object_sources=args.sources,
        render_mode="rgb_array" if args.rgb else None,
    )

    print(f"\n=== PickAnything randomization check ({obs_mode} obs) ===")
    print(f"sources = {args.sources}\n")
    print("Unseeded resets (each should differ):\n")
    for ep in range(args.num_episodes):
        obs, _ = env.reset()
        env_ = env.unwrapped
        obj = env_.obj
        goal = env_.goal_site
        src = env_.object_sources[0]
        z = float(env_.object_zs[0])
        print(
            f"  ep {ep}: source={src:<10} obj_pos={obj.pose.p[0].cpu().numpy().round(3)} "
            f"goal_pos={goal.pose.p[0].cpu().numpy().round(3)} rest_z={z:.4f}"
        )
        if args.rgb:
            frame = env.render()
            if frame is not None:
                img = np.asarray(frame)
                if img.ndim == 4:
                    img = img[0]
                print(f"        rgb frame shape={img.shape} mean={img.mean():.1f}")

    print("\nReproducibility (same seed -> same result):")
    env.reset(seed=args.seed)
    pos1 = env.unwrapped.obj.pose.p[0].cpu().numpy().copy()
    src1 = env.unwrapped.object_sources[0]
    env.reset(seed=args.seed)
    pos2 = env.unwrapped.obj.pose.p[0].cpu().numpy().copy()
    src2 = env.unwrapped.object_sources[0]
    same = np.allclose(pos1, pos2) and src1 == src2
    print(
        f"  @seed={args.seed}: run1 source={src1} pos={pos1.round(3)}\n"
        f"            run2 source={src2} pos={pos2.round(3)}\n"
        f"  identical={same}"
    )

    env.close()
    print("\nOK." if same else "\nWARNING: reproducibility check failed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
