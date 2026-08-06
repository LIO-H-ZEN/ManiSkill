import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode


import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PushCube-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "none"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    render_backend: Annotated[str, tyro.conf.arg(aliases=["-rb"])] = "gpu"
    """Which render backend to use. Can be 'gpu', 'cpu', 'none'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, list[int]]], tyro.conf.arg(aliases=["-s"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

    object_sources: Optional[List[str]] = None
    """PickAnything only: object source aliases to mix each reconfigure, e.g.
    `--object-sources cube ycb` or `--object-sources cube ycb interndata`.
    Valid: cube, ycb, interndata. Defaults to the env default (cube+ycb+
    interndata). interndata downloads meshes on demand from a gated HF dataset
    (InternRobotics/InternData-A1); needs `huggingface-cli login` + license
    acceptance."""

    table_randomizer: Optional[List[str]] = None
    """PickAnything only: table surface randomizer alias, or a list of them to
    mix per reconfigure. `wood` = fixed PickCube wood table; `texture` = real
    InternDataAssets table-surface texture + random friction; `procedural` =
    random PBR color/metallic/roughness. None uses the env default
    (wood + texture mix). `texture` downloads textures on demand (gated HF)."""

    floor_randomizer: Optional[str] = None
    """PickAnything only: floor (ground) randomizer alias. `texture` (default) =
    InternDataAssets floor textures (floor_textures + background_textures);
    `grid` = checkered grid (no download). Independent of the table."""

    clutter: int = 0
    """PickAnything only: number of distractor objects per env (clutter). 0
    (default) = no distractors. Distractors reuse the object-sources pool and
    are placed on the table avoiding the target. Orthogonal to all other axes."""

    num_episodes: int = 1
    """Number of episodes to run in non-human render mode (each reconfigures,
    re-randomizing PickAnything object/table/lighting). In human render mode the
    demo runs indefinitely, resetting to a fresh episode on the same table each
    time an episode ends (use a different -s seed to see a different table)."""

def main(args: Args):
    if args.render_mode == "none":
        args.render_mode = None
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
        if len(env_kwargs["robot_uids"]) == 1:
            env_kwargs["robot_uids"] = env_kwargs["robot_uids"][0]
    if args.object_sources is not None:
        env_kwargs["object_sources"] = args.object_sources
    if args.table_randomizer is not None:
        env_kwargs["table_randomizer"] = args.table_randomizer
    if args.floor_randomizer is not None:
        env_kwargs["floor_randomizer"] = args.floor_randomizer
    if args.clutter and args.clutter > 0:
        env_kwargs["clutter"] = args.clutter
    env: BaseEnv = gym.make(
        args.env_id,
        **env_kwargs
    )
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=gym_utils.find_max_episode_steps_value(env))

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    if args.seed is not None and env.action_space is not None:
            env.action_space.seed(args.seed[0])
    if args.render_mode == "human":
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    human = args.render_mode == "human"
    ep = 0
    while True:
        if verbose:
            uw = env.unwrapped
            bits = []
            choice = getattr(uw, "table_randomizer_choice", None)
            if choice is not None:
                bits.append(f"table={choice}")
            tex = getattr(uw, "table_texture", None)
            if tex is not None:
                bits.append(f"tex={tex}")
            mtype = getattr(uw, "table_material_type", None)
            if mtype is not None:
                bits.append(f"mtype={mtype}")
            fric = getattr(uw, "table_friction", None)
            if fric is not None:
                bits.append(f"friction={fric}")
            ftex = getattr(uw, "floor_texture", None)
            if ftex is not None:
                bits.append(f"floor={ftex}")
            if bits:
                print(f"[ep {ep}] " + " ".join(bits))
        while True:
            action = env.action_space.sample() if env.action_space is not None else None
            obs, reward, terminated, truncated, info = env.step(action)
            if verbose:
                print("reward", reward)
                print("terminated", terminated)
                print("truncated", truncated)
                print("info", info)
            if human:
                env.render()
            if (terminated | truncated).any():
                break
        ep += 1
        if not human and ep >= args.num_episodes:
            break
        if human:
            # keep running forever: fresh episode on the same table
            env.reset()
        else:
            # reconfigure to re-randomize PickAnything object/table/lighting
            env.reset(options=dict(reconfigure=True))
    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
