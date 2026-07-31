"""Verify that PickAnything randomizes across episodes and is reproducible.

Run from the repo root:

    python -m mani_skill.examples.verify_pick_anything
    python -m mani_skill.examples.verify_pick_anything --rgb   # also grab a camera frame

This does NOT need any asset download --- v1 is fully procedural.
"""

import gymnasium as gym
import numpy as np
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    rgb: bool = False
    """Also render one RGB camera frame per episode to confirm rendering works."""
    num_episodes: int = 6


def main(args: Args):
    import mani_skill.envs  # noqa: register envs

    obs_mode = "rgb" if args.rgb else "state"
    env = gym.make(
        "PickAnything-v1",
        obs_mode=obs_mode,
        num_envs=1,
        render_mode="rgb_array" if args.rgb else None,
    )

    print(f"\n=== PickAnything randomization check ({obs_mode} obs) ===")
    print("Unseeded resets (each should differ):\n")
    for ep in range(args.num_episodes):
        obs, _ = env.reset()
        obj = env.unwrapped.obj
        goal = env.unwrapped.goal_site
        hs = float(env.unwrapped.object_half_sizes[0])
        print(
            f"  ep {ep}: obj_pos={obj.pose.p[0].cpu().numpy().round(3)} "
            f"goal_pos={goal.pose.p[0].cpu().numpy().round(3)} "
            f"half_size={hs:.4f}"
        )
        if args.rgb:
            frame = env.render()
            if frame is not None:
                import numpy as _np
                img = _np.asarray(frame)
                if img.ndim == 4:
                    img = img[0]
                print(f"        rgb frame shape={img.shape} mean={img.mean():.1f}")

    print("\nReproducibility (same seed -> same result):")
    env.reset(seed=123)
    pos1 = env.unwrapped.obj.pose.p[0].cpu().numpy().copy()
    env.reset(seed=123)
    pos2 = env.unwrapped.obj.pose.p[0].cpu().numpy().copy()
    same = np.allclose(pos1, pos2)
    print(f"  obj pos @seed=123 run1={pos1.round(3)} run2={pos2.round(3)} identical={same}")

    env.close()
    print("\nOK." if same else "\nWARNING: reproducibility check failed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
