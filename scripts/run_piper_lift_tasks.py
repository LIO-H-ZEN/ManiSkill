#!/usr/bin/env python3
"""Launch the PIPER LiftCube and LiftAnything environments from one command."""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import pathlib
import re
from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mani_skill.envs  # noqa: F401 - imports register the environments
from mani_skill.envs.tasks.pick_anything.episode_specs import EpisodeSpec
from mani_skill.utils import common
from mani_skill.utils.wrappers import RecordEpisode

CAMERA_UIDS = ("base_camera", "wrist_camera", "side_camera")
CAMERA_LABELS = {
    "base_camera": "Base camera",
    "wrist_camera": "Wrist camera",
    "side_camera": "Side camera",
}
ENV_IDS = {
    "liftcube": "LiftCubePiper-v1",
    "liftanything": "LiftAnythingPiper-v1",
}
DEFAULT_TASK_PROMPTS = {
    "liftcube": "Pick up the red cube.",
}
DEFAULT_VIDEO_DIR = pathlib.Path(__file__).resolve().parents[2] / "piper_lift_videos"
OVERLAY_FONT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "mani_skill/utils/visualization/UbuntuSansMono-Regular.ttf"
)
OVERLAY_BACKGROUND = (12, 16, 22, 185)
OVERLAY_TEXT = (255, 255, 255, 255)


@functools.lru_cache(maxsize=None)
def _overlay_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(OVERLAY_FONT_PATH), size=size)


def _text_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont
) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    words = text.split()
    if not words:
        raise ValueError("overlay text must not be empty")

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if _text_width(draw, candidate, font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        while _text_width(draw, word, font) > max_width:
            split_at = len(word) - 1
            while split_at > 0 and _text_width(draw, word[:split_at], font) > max_width:
                split_at -= 1
            if split_at == 0:
                raise ValueError("overlay width is too small for the selected font")
            lines.append(word[:split_at])
            word = word[split_at:]
        current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def _draw_text_badge(
    image: np.ndarray,
    text: str,
    *,
    font_size: int,
    placement: str,
    margin: int,
    horizontal_padding: int,
    vertical_padding: int,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise RuntimeError(
            "Video frame violates the text overlay contract: "
            f"shape={image.shape}, dtype={image.dtype}"
        )
    if placement not in {"top-left", "bottom-center"}:
        raise ValueError(f"Unsupported text placement: {placement}")

    canvas = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _overlay_font(font_size)
    max_text_width = image.shape[1] - 2 * (margin + horizontal_padding)
    wrapped_text = _wrap_text(draw, text, font, max_text_width)
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=2)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    badge_width = text_width + 2 * horizontal_padding
    badge_height = text_height + 2 * vertical_padding

    if placement == "top-left":
        left = margin
        top = margin
    else:
        left = (image.shape[1] - badge_width) // 2
        top = image.shape[0] - margin - badge_height
    right = left + badge_width
    bottom = top + badge_height
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=min(8, badge_height // 2),
        fill=OVERLAY_BACKGROUND,
    )
    draw.multiline_text(
        (left + horizontal_padding - bbox[0], top + vertical_padding - bbox[1]),
        wrapped_text,
        font=font,
        fill=OVERLAY_TEXT,
        spacing=2,
        align="center" if placement == "bottom-center" else "left",
    )
    return np.asarray(Image.alpha_composite(canvas, overlay).convert("RGB")).copy()


def _camera_triptych(sensor_images: dict[str, dict[str, Any]]) -> np.ndarray:
    panels = []
    for camera_uid in CAMERA_UIDS:
        try:
            image = sensor_images[camera_uid]["rgb"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                f"Video recording is missing RGB data for {camera_uid}"
            ) from error
        image = np.asarray(common.to_numpy(image))
        if image.shape == (1, 224, 224, 3):
            image = image[0]
        if image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise RuntimeError(
                f"{camera_uid} violates the video image contract: "
                f"shape={image.shape}, dtype={image.dtype}"
            )
        panels.append(
            _draw_text_badge(
                image,
                CAMERA_LABELS[camera_uid],
                font_size=13,
                placement="top-left",
                margin=8,
                horizontal_padding=6,
                vertical_padding=4,
            )
        )
    return np.concatenate(panels, axis=1)


def _with_task_prompt(image: np.ndarray, task_prompt: str) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise RuntimeError(
            "Video frame violates the prompt overlay contract: "
            f"shape={image.shape}, dtype={image.dtype}"
        )
    prompt = task_prompt.strip()
    if not prompt:
        raise ValueError("task prompt must not be empty")
    return _draw_text_badge(
        image,
        prompt,
        font_size=16,
        placement="bottom-center",
        margin=10,
        horizontal_padding=10,
        vertical_padding=6,
    )


TaskPrompt = str | Callable[[], str]


def _humanize_object_name(actor_name: str) -> str:
    """Convert a source actor name into a concise visual object label."""
    name = re.sub(r"-\d+$", "", actor_name)
    if name == "cube":
        return "cube"
    if name.startswith("cube-"):
        return "cube"
    if name.startswith("ycb-"):
        name = re.sub(r"^\d+_", "", name.removeprefix("ycb-"))
    elif name.startswith("interndata-"):
        name = name.removeprefix("interndata-")
        name = re.sub(r"^(?:omniobject3d|google_scan|phocal)-", "", name)
        name = re.sub(r"_\d+$", "", name)
    else:
        raise RuntimeError(
            f"Cannot infer a task prompt from target actor name {actor_name!r}; "
            "pass --task-prompt explicitly"
        )
    label = name.replace("_", " ").replace("-", " ").strip()
    if not label:
        raise RuntimeError(
            f"Cannot infer a task prompt from target actor name {actor_name!r}; "
            "pass --task-prompt explicitly"
        )
    return label


def _default_task_prompt(task: str, base_env) -> str:
    if task == "liftcube":
        return DEFAULT_TASK_PROMPTS[task]
    object_names = getattr(base_env.unwrapped, "object_names", None)
    if not object_names:
        raise RuntimeError(
            "LiftAnything did not expose the sampled target object name; "
            "pass --task-prompt explicitly"
        )
    return f"Pick up the {_humanize_object_name(str(object_names[0]))}."


def _resolve_task_prompt(task_prompt: TaskPrompt) -> str:
    return task_prompt() if callable(task_prompt) else task_prompt


class PromptedRecordEpisode(RecordEpisode):
    """Record the human render camera with an in-frame task-prompt overlay."""

    def __init__(self, *args, task_prompt: TaskPrompt, **kwargs):
        self.task_prompt = task_prompt
        super().__init__(*args, **kwargs)

    def capture_image(self, infos=None):
        return _with_task_prompt(
            super().capture_image(infos), _resolve_task_prompt(self.task_prompt)
        )


class ThreeCameraRecordEpisode(PromptedRecordEpisode):
    """Record a labeled base/wrist/side camera triptych for every control step."""

    def capture_image(self, infos=None):
        return _with_task_prompt(
            _camera_triptych(self.base_env.get_sensor_images()),
            _resolve_task_prompt(self.task_prompt),
        )


def _video_recorder_class(video_view: str):
    if video_view == "triptych":
        return ThreeCameraRecordEpisode
    if video_view == "third-person":
        return PromptedRecordEpisode
    raise ValueError(f"Unsupported video view: {video_view}")


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start the PIPER LiftCube/LiftAnything tasks with their locked "
            "three-camera and pd_joint_pos contract."
        )
    )
    parser.add_argument(
        "--task",
        choices=("liftcube", "liftanything", "both"),
        default="both",
        help="Task to run. The default runs one episode of each task.",
    )
    parser.add_argument(
        "--mode",
        choices=("reset", "expert"),
        default="expert",
        help="Only reset and inspect the environment, or run its MPLib expert.",
    )
    parser.add_argument("--seed", type=_non_negative_int, default=0)
    parser.add_argument(
        "--num-envs",
        type=_positive_int,
        default=1,
        help=(
            "Number of parallel environments. Values above one are supported "
            "only in reset mode; the MPLib experts are single-environment."
        ),
    )
    parser.add_argument(
        "--render-backend",
        default="cuda:0",
        help="SAPIEN render backend, for example cuda:0 on a GPU worker.",
    )
    parser.add_argument(
        "--object-source",
        choices=("cube", "ycb", "interndata"),
        help=(
            "LiftAnything object source. Defaults to cube so the first run does "
            "not require YCB or InternData assets."
        ),
    )
    parser.add_argument(
        "--table-randomizer",
        choices=("wood", "texture", "procedural"),
        default="wood",
        help=(
            "LiftAnything table surface. Use procedural or texture to make the "
            "table axis visibly change mid-episode."
        ),
    )
    parser.add_argument(
        "--floor-randomizer",
        choices=("grid", "texture"),
        default="grid",
        help="LiftAnything floor appearance.",
    )
    parser.add_argument(
        "--clutter",
        help=(
            "LiftAnything distractors: a fixed count such as 3, a per-episode "
            "range such as random_2_5, or 0/off to disable."
        ),
    )
    parser.add_argument(
        "--clutter-sources",
        nargs="+",
        choices=("cube", "ycb", "interndata"),
        help=(
            "Independent LiftAnything distractor source pool. For example, "
            "use --object-source cube --clutter-sources interndata. By default "
            "clutter reuses the target object source."
        ),
    )
    parser.add_argument(
        "--domain-rand-freq",
        type=_non_negative_int,
        default=25,
        help="LiftAnything mid-episode randomization cadence; 0 disables it.",
    )
    parser.add_argument(
        "--domain-rand-axes",
        nargs="*",
        choices=("lighting", "table", "clutter"),
        help=(
            "LiftAnything axes changed at the configured cadence. Omit the "
            "option for all axes; pass it with no values to disable every axis."
        ),
    )
    parser.add_argument(
        "--episode-spec",
        type=pathlib.Path,
        help="JSON file containing one EpisodeSpec or an episodes list.",
    )
    parser.add_argument(
        "--episode-index",
        type=_non_negative_int,
        default=0,
        help="EpisodeSpec index when --episode-spec contains multiple episodes.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Ask the MPLib expert to render its human-view visualization.",
    )
    parser.add_argument(
        "--video-dir",
        type=pathlib.Path,
        default=DEFAULT_VIDEO_DIR,
        help=(
            "Directory under which each expert run saves MP4 files. The default "
            "is outside the Git repository."
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=_positive_int,
        default=20,
        help="Frames per second for saved expert videos.",
    )
    parser.add_argument(
        "--video-view",
        choices=("triptych", "third-person"),
        default="triptych",
        help="Record the three sensor cameras together or the render camera.",
    )
    parser.add_argument(
        "--task-prompt",
        help=(
            "Text displayed on every video frame. If omitted, the launcher "
            "names the sampled target object automatically."
        ),
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Run the expert without saving MP4 videos.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.task == "liftcube" and args.episode_spec is not None:
        raise ValueError("--episode-spec is only valid for LiftAnything")
    if args.task == "liftcube" and args.object_source is not None:
        raise ValueError("--object-source is only valid for LiftAnything")
    if args.task == "liftcube" and args.clutter_sources is not None:
        raise ValueError("--clutter-sources is only valid for LiftAnything")
    if args.episode_spec is not None and args.object_source is not None:
        raise ValueError("--episode-spec and --object-source are mutually exclusive")
    if args.clutter_sources is not None and (
        args.clutter is None or args.clutter.strip().lower() in {"", "0", "none", "off"}
    ):
        raise ValueError("--clutter-sources requires enabled --clutter")
    if args.task_prompt is not None and not args.task_prompt.strip():
        raise ValueError("--task-prompt must not be empty")
    if args.mode == "reset" and args.vis:
        raise ValueError("--vis requires --mode expert")
    if args.num_envs > 1 and args.mode == "expert":
        raise ValueError("--num-envs > 1 is only supported with --mode reset")
    if args.num_envs > 1 and args.episode_spec is not None:
        raise ValueError("--episode-spec requires --num-envs 1")


def _load_episode_spec(path: pathlib.Path, index: int) -> EpisodeSpec:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "episodes" in payload:
        rows = payload["episodes"]
    elif isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and "stable_episode_id" in payload:
        rows = [payload]
    else:
        raise ValueError(
            "EpisodeSpec JSON must be one spec, a list, or an object with an episodes list"
        )
    if not isinstance(rows, list) or not rows:
        raise ValueError("EpisodeSpec JSON contains no episodes")
    if index >= len(rows):
        raise IndexError(
            f"episode-index {index} is outside the available range [0, {len(rows) - 1}]"
        )
    if not isinstance(rows[index], dict):
        raise ValueError(f"EpisodeSpec at index {index} is not a JSON object")
    return EpisodeSpec.from_dict(rows[index])


def _environment_kwargs(
    task: str, args: argparse.Namespace
) -> tuple[str, dict[str, Any]]:
    env_id = ENV_IDS[task]
    kwargs: dict[str, Any] = {
        "robot_uids": "piper_wristcam",
        "obs_mode": "rgb",
        "control_mode": "pd_joint_pos",
        "sim_backend": "physx_cuda" if args.num_envs > 1 else "physx_cpu",
        "num_envs": args.num_envs,
        "max_episode_steps": 100,
        "render_backend": args.render_backend,
    }
    if args.mode == "expert" and not args.no_video:
        kwargs["render_mode"] = (
            "sensors" if args.video_view == "triptych" else "rgb_array"
        )
    if task == "liftanything":
        kwargs["domain_rand_freq"] = args.domain_rand_freq
        if args.domain_rand_axes is not None:
            kwargs["domain_rand_axes"] = tuple(args.domain_rand_axes)
        if args.clutter is not None:
            kwargs["clutter"] = args.clutter
        if args.clutter_sources is not None:
            kwargs["clutter_sources"] = tuple(args.clutter_sources)
        if args.episode_spec is not None:
            spec = _load_episode_spec(args.episode_spec, args.episode_index)
            kwargs["episode_spec"] = spec.to_dict()
        else:
            kwargs["object_sources"] = (args.object_source or "cube",)
            kwargs["table_randomizer"] = args.table_randomizer
            kwargs["floor_randomizer"] = args.floor_randomizer
    return env_id, kwargs


def _camera_contract(
    observation: dict[str, Any], num_envs: int = 1
) -> dict[str, list[int]]:
    try:
        sensor_data = observation["sensor_data"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("RGB observation is missing sensor_data") from error
    shapes: dict[str, list[int]] = {}
    for camera_uid in CAMERA_UIDS:
        try:
            image = sensor_data[camera_uid]["rgb"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(f"RGB observation is missing {camera_uid}") from error
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        expected_shape = (num_envs, 224, 224, 3)
        if image.shape != expected_shape or image.dtype != np.uint8:
            raise RuntimeError(
                f"{camera_uid} violates the image contract: "
                f"shape={image.shape}, dtype={image.dtype}"
            )
        shapes[camera_uid] = list(image.shape)
    return shapes


def _as_bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def _run_expert(task: str, env, *, seed: int, vis: bool) -> dict[str, Any]:
    if task == "liftcube":
        from mani_skill.examples.motionplanning.piper.solutions.lift_cube import (
            solve,
        )

        transition = solve(env, seed=seed, vis=vis)
        if transition is None:
            raise RuntimeError("LiftCube expert returned no transition")
        success = _as_bool(transition[4]["success"])
        if not success:
            raise RuntimeError("LiftCube expert finished without task success")
        return {"success": True, "reason": "accepted"}

    from mani_skill.examples.motionplanning.piper.solutions.lift_anything import (
        solve,
    )

    result = solve(env, seed=seed, vis=vis)
    if not result.success:
        raise RuntimeError(
            "LiftAnything expert failed after "
            f"{result.attempted_candidates} candidates: {result.reason}"
        )
    return {
        "success": True,
        "reason": result.reason,
        "candidate_id": result.candidate_id,
        "attempted_candidates": result.attempted_candidates,
    }


def run_task(task: str, args: argparse.Namespace) -> dict[str, Any]:
    env_id, kwargs = _environment_kwargs(task, args)
    print(f"\nStarting {env_id} with seed={args.seed}", flush=True)
    print(json.dumps({"environment": env_id, "config": kwargs}, default=str, indent=2))
    base_env = gym.make(env_id, **kwargs)
    task_prompt: TaskPrompt
    if args.task_prompt is not None:
        task_prompt = args.task_prompt
    elif task == "liftcube":
        task_prompt = DEFAULT_TASK_PROMPTS[task]
    else:
        task_prompt = lambda: _default_task_prompt(task, base_env)
    env = base_env
    video_output_dir: pathlib.Path | None = None
    if args.mode == "expert" and not args.no_video:
        video_output_dir = args.video_run_dir / env_id
        recorder_class = _video_recorder_class(args.video_view)
        env = recorder_class(
            env,
            output_dir=str(video_output_dir),
            save_trajectory=False,
            save_video=True,
            info_on_video=False,
            save_on_reset=True,
            video_fps=args.video_fps,
            avoid_overwriting_video=True,
            task_prompt=task_prompt,
        )
    try:
        observation, _ = env.reset(
            seed=args.seed,
            options={"reconfigure": True},
        )
        camera_shapes = _camera_contract(observation, args.num_envs)
        summary: dict[str, Any] = {
            "environment": env_id,
            "seed": args.seed,
            "camera_shapes": camera_shapes,
            "mode": args.mode,
            "task_prompt": _resolve_task_prompt(task_prompt),
        }
        if args.mode == "expert":
            summary["expert"] = _run_expert(
                task,
                env,
                seed=args.seed,
                vis=args.vis,
            )
    finally:
        env.close()
    if video_output_dir is not None:
        summary["video_view"] = args.video_view
        summary["video_files"] = [
            str(path) for path in sorted(video_output_dir.glob("*.mp4"))
        ]
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        if args.mode == "expert" and not args.no_video:
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            args.video_run_dir = args.video_dir.expanduser().resolve() / run_id
            print(f"Videos will be saved under {args.video_run_dir}", flush=True)
        tasks = ("liftcube", "liftanything") if args.task == "both" else (args.task,)
        for task in tasks:
            run_task(task, args)
    except (FileNotFoundError, IndexError, ValueError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
