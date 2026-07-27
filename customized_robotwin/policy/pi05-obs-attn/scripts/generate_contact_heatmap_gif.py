#!/usr/bin/env python
"""Generate standalone contact-attention GIFs from RoboPRO episodes and caches.

No web server is required. Each frame shows three rows:
  1. input, beta, obstacle contact, obstacle GT
  2. obstacle grid, target grasp contact, target GT, target grid
  3. destination mask, destination GT, destination grid
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys

import cv2
import h5py
import matplotlib as mpl
import numpy as np
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import beta_geometry as bg  # noqa: E402

SIZE = 224
CLUTTER_LEVELS = ("clean", *(f"d{density}" for density in range(6, 16)))
_COOLWARM_BGR = (
    mpl.colormaps["coolwarm"](np.linspace(0, 1, 256))[:, :3] * 255
).astype(np.uint8)[:, ::-1]


def episode_number(filename: str) -> int:
    digits = "".join(character for character in Path(filename).stem if character.isdigit())
    return int(digits) if digits else -1


def destination_ids(cfg_dir: Path, filename: str) -> np.ndarray | None:
    episode_idx = episode_number(filename)
    masking_path = cfg_dir / "masking" / f"episode{episode_idx}.json"
    if masking_path.is_file():
        with masking_path.open() as file:
            data = json.load(file)
        by_stage = {
            int(stage["stage"]): int(stage["bin"]["bin_id"])
            for stage in data.get("stages", [])
            if (stage.get("bin") or {}).get("bin_id") is not None
        }
        frame_stage = data.get("frame_stage")
        if by_stage and frame_stage:
            return np.asarray([by_stage.get(int(stage), -1) for stage in frame_stage], dtype=np.int64)

    scene_path = cfg_dir / "scene_info.json"
    if scene_path.is_file():
        with scene_path.open() as file:
            episode = json.load(file).get(f"episode_{episode_idx}", {})
        destination_id = (episode.get("role_names") or {}).get("destination_id")
        if destination_id is not None:
            return np.asarray([int(destination_id)], dtype=np.int64)
    return None


def discover(data_root: str, cache_subdir: str, beta_root: str | None = None) -> list[dict]:
    beta_base = Path(beta_root).expanduser() if beta_root else None
    episodes = []
    for dirpath, dirnames, filenames in os.walk(data_root):
        dirnames[:] = [directory for directory in dirnames if directory not in {".cache", "video", "scene"}]
        if Path(dirpath).name != "data":
            continue
        cfg_dir = Path(dirpath).parent
        # With beta_root set, data_root is read-only and caches live under a mirrored
        # <beta_root>/<domain>/<task>/<config>/<cache_subdir>/ layout (matching how
        # precompute_beta_weights.py / the converter write them); otherwise they sit
        # next to the HDF5s in <cfg_dir>/<cache_subdir>/.
        cache_dir = beta_base / cfg_dir.relative_to(data_root) / cache_subdir if beta_base else cfg_dir / cache_subdir
        for filename in filenames:
            if not filename.endswith(".hdf5"):
                continue
            path = Path(dirpath) / filename
            cache = cache_dir / filename.replace(".hdf5", ".npz")
            episodes.append(
                {
                    "path": str(path),
                    "cache": str(cache) if cache.is_file() else None,
                    "label": f"{cfg_dir.relative_to(data_root)}/{filename}",
                    "destination_ids": destination_ids(cfg_dir, filename),
                }
            )
    episodes.sort(key=lambda episode: episode["label"])
    for index, episode in enumerate(episodes):
        episode["id"] = index
    return episodes


def load_cache(path: str | None) -> dict:
    if path is None:
        return {}
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def decode(dataset, frame: int, flag=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(dataset[frame], dtype=np.uint8), flag)


def resize_with_pad(array: np.ndarray, interpolation: int) -> np.ndarray:
    height, width = array.shape[:2]
    ratio = max(height / SIZE, width / SIZE)
    resized_h, resized_w = max(1, int(height / ratio)), max(1, int(width / ratio))
    resized = cv2.resize(array, (resized_w, resized_h), interpolation=interpolation)
    out = np.zeros((SIZE, SIZE, *array.shape[2:]), dtype=array.dtype)
    y0, x0 = (SIZE - resized_h) // 2, (SIZE - resized_w) // 2
    out[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return out


def bbox_frame(h5: h5py.File, frame: int):
    def read_group(group):
        ids = group["id"][frame]
        if all(key in group for key in ("obb_center", "obb_half", "obb_quat")):
            return (
                ids,
                group["obb_center"][frame],
                group["obb_half"][frame],
                group["obb_quat"][frame],
            )
        mins, maxs = group["aabb_min"][frame], group["aabb_max"][frame]
        center, half = 0.5 * (mins + maxs), 0.5 * (maxs - mins)
        quat = np.zeros((*center.shape[:-1], 4), dtype=np.float32)
        quat[..., 0] = 1.0
        return ids, center, half, quat

    tables = [read_group(h5["actor_bbox"])]
    if "link_bbox" in h5 and "id" in h5["link_bbox"]:
        tables.append(read_group(h5["link_bbox"]))
    return tuple(np.concatenate([table[index] for table in tables]) for index in range(4))


def project_anchor(local, center, half, quat, extrinsic, intrinsic):
    world = bg.obb_normalized_to_world(local, center, half, quat).reshape(1, 3)
    uv, depth = bg.project_world_to_image(world, extrinsic, intrinsic)
    return uv[0] if depth[0] > 1e-6 else None


def contact_disk(seg, actor_id: int, uv, radius: int, max_snap: float = 48.0):
    if uv is None:
        return None
    ys, xs = np.nonzero(seg == actor_id)
    if xs.size == 0:
        return None
    distance2 = (xs - uv[0]) ** 2 + (ys - uv[1]) ** 2
    nearest = int(np.argmin(distance2))
    if distance2[nearest] > max_snap**2:
        return None
    cx, cy = int(xs[nearest]), int(ys[nearest])
    yy, xx = np.ogrid[: seg.shape[0], : seg.shape[1]]
    return (((xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2) & (seg == actor_id)).astype(
        np.float32
    )


def gaussian_heatmap(mask: np.ndarray, sigma: float) -> np.ndarray:
    kernel_size = 2 * int(np.ceil(2 * sigma)) + 1
    axis = np.arange(kernel_size, dtype=np.float32) - (kernel_size - 1) / 2
    kernel = np.exp(-0.5 * (axis / sigma) ** 2)
    kernel /= kernel.sum()
    heat = cv2.sepFilter2D(
        mask.astype(np.float32),
        cv2.CV_32F,
        kernel,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    return heat / heat.sum() if heat.sum() > 0 else heat


def patchify(heatmap: np.ndarray, patch_size: int) -> np.ndarray:
    grid_h, grid_w = SIZE // patch_size, SIZE // patch_size
    pooled = heatmap.reshape(grid_h, patch_size, grid_w, patch_size).sum(axis=(1, 3))
    return pooled / pooled.sum() if pooled.sum() > 0 else pooled


def colorize_heat(heatmap: np.ndarray) -> np.ndarray:
    normalized = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
    return cv2.applyColorMap(
        (np.clip(normalized, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET
    )


def colorize_beta(beta: np.ndarray, gain: float) -> np.ndarray:
    position = np.where(
        beta <= 1,
        0.5 * beta,
        0.5 + 0.5 * np.clip((beta - 1) / max(gain, 1e-6), 0, 1),
    )
    return _COOLWARM_BGR[(np.clip(position, 0, 1) * 255).astype(np.uint8)]


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.58) -> np.ndarray:
    return cv2.addWeighted(image, 1 - alpha, colorize_heat(heatmap), alpha, 0)


def label(image: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(image, (0, 0), (image.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(
        image, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA
    )
    return image


def role_contact(cache: dict, prefix: str, frame: int):
    ids = cache.get(f"{prefix}_contact_id")
    local = cache.get(f"{prefix}_contact_local")
    valid = cache.get(f"{prefix}_contact_valid")
    if ids is None or local is None or valid is None or frame >= len(ids):
        return -1, None, False
    local_frame = local[frame] if local.ndim == 2 else local
    valid_frame = bool(valid[frame]) if valid.ndim else bool(valid)
    return int(ids[frame]), local_frame, valid_frame


def render_frame(
    h5: h5py.File,
    episode: dict,
    cache: dict,
    frame: int,
    *,
    camera_name: str,
    sigma: float,
    radius: int,
    patch_size: int,
) -> Image.Image:
    camera = h5[f"observation/{camera_name}"]
    rgb = resize_with_pad(decode(camera["rgb"], frame, cv2.IMREAD_COLOR), cv2.INTER_LINEAR)
    seg = decode(camera["actor_segmentation"], frame)
    extrinsic, intrinsic = camera["extrinsic_cv"][frame], camera["intrinsic_cv"][frame]
    ids, centers, halves, quats = bbox_frame(h5, frame)
    id_to_col = {int(actor_id): col for col, actor_id in enumerate(ids)}

    obstacle_seed = np.zeros(seg.shape, dtype=np.float32)
    beta_field = np.ones(seg.shape, dtype=np.float32)
    localized = fallback = 0
    object_ids = cache.get("obj_ids", np.zeros(0, dtype=np.int64))
    obstacle_local = cache.get("obstacle_contact_local")
    obstacle_valid = cache.get("obstacle_contact_valid")
    beta = cache.get("contact_beta", cache.get("beta"))
    for cache_col, actor_id_value in enumerate(object_ids):
        actor_id = int(actor_id_value)
        actor_mask = (seg == actor_id).astype(np.float32)
        if not actor_mask.any():
            continue
        contact = None
        valid = (
            obstacle_local is not None
            and frame < obstacle_local.shape[0]
            and (obstacle_valid is None or bool(obstacle_valid[frame, cache_col]))
        )
        bbox_col = id_to_col.get(actor_id)
        if valid and bbox_col is not None:
            uv = project_anchor(
                obstacle_local[frame, cache_col],
                centers[bbox_col],
                halves[bbox_col],
                quats[bbox_col],
                extrinsic,
                intrinsic,
            )
            contact = contact_disk(seg, actor_id, uv, radius)
        if contact is None:
            contact = actor_mask
            fallback += 1
        else:
            localized += 1
        obstacle_seed = np.maximum(obstacle_seed, contact)
        if beta is not None and frame < beta.shape[0]:
            beta_field[seg == actor_id] = float(beta[frame, cache_col])

    target_seed = np.zeros(seg.shape, dtype=np.float32)
    target_id, target_local, target_valid = role_contact(cache, "target", frame)
    target_col = id_to_col.get(target_id)
    if target_valid and target_col is not None:
        uv = project_anchor(
            target_local,
            centers[target_col],
            halves[target_col],
            quats[target_col],
            extrinsic,
            intrinsic,
        )
        contact = contact_disk(seg, target_id, uv, radius)
        target_seed = contact if contact is not None else (seg == target_id).astype(np.float32)

    destination_seed = np.zeros(seg.shape, dtype=np.float32)
    destinations = episode["destination_ids"]
    if destinations is not None and destinations.size:
        destination_id = int(destinations[min(frame, destinations.shape[0] - 1)])
        if destination_id >= 0:
            destination_seed = (seg == destination_id).astype(np.float32)

    obstacle_seed = resize_with_pad(obstacle_seed, cv2.INTER_NEAREST)
    beta_field = resize_with_pad(beta_field, cv2.INTER_NEAREST)
    target_seed = resize_with_pad(target_seed, cv2.INTER_NEAREST)
    destination_seed = resize_with_pad(destination_seed, cv2.INTER_NEAREST)
    obstacle_heat = gaussian_heatmap(obstacle_seed * beta_field, sigma)
    target_heat = gaussian_heatmap(target_seed, sigma)
    destination_heat = gaussian_heatmap(destination_seed, sigma)

    obstacle_grid = patchify(obstacle_heat, patch_size)
    target_grid = patchify(target_heat, patch_size)
    destination_grid = patchify(destination_heat, patch_size)

    def grid_image(grid):
        return cv2.resize(grid, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

    gain = float(cache["gain"]) if "gain" in cache else 3.0
    panels = [
        label(rgb.copy(), "input"),
        label(colorize_beta(beta_field, gain), f"beta [{beta_field.min():.2f},{beta_field.max():.2f}]"),
        label(overlay(rgb, obstacle_seed), f"obstacle contact ({localized} local/{fallback} fallback)"),
        label(overlay(rgb, obstacle_heat), "obstacle GT heatmap"),
        label(colorize_heat(grid_image(obstacle_grid)), f"obstacle token grid {obstacle_grid.shape}"),
        label(overlay(rgb, target_seed), "target grasp contact"),
        label(overlay(rgb, target_heat), "target GT heatmap"),
        label(colorize_heat(grid_image(target_grid)), f"target token grid {target_grid.shape}"),
        label(overlay(rgb, destination_seed), "destination mask"),
        label(overlay(rgb, destination_heat), "destination GT heatmap"),
        label(colorize_heat(grid_image(destination_grid)), f"destination token grid {destination_grid.shape}"),
    ]
    columns = 4
    panels.extend([np.full_like(panels[0], 30)] * ((-len(panels)) % columns))
    rows = [
        np.hstack(
            [
                np.pad(panel, ((0, 0), (0, 3), (0, 0)), constant_values=30)
                for panel in panels[start : start + columns]
            ]
        )
        for start in range(0, len(panels), columns)
    ]
    separator = np.full((3, rows[0].shape[1], 3), 30, dtype=np.uint8)
    strip = np.vstack(
        [item for index, row in enumerate(rows) for item in ((separator,) if index else ()) + (row,)]
    )
    ok, png = cv2.imencode(".png", strip)
    if not ok:
        raise RuntimeError("failed to encode visualization frame")
    return Image.open(io.BytesIO(png.tobytes())).convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("generate contact-oriented attention GIFs")
    parser.add_argument("--data-root", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--episode",
        type=int,
        default=None,
        help="episode number within every selected task (default: 0)",
    )
    selection.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help="episode numbers within every selected task, e.g. 1 4 7",
    )
    selection.add_argument(
        "--episode-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="inclusive episode-number range within every selected task",
    )
    selection.add_argument("--all-episodes", action="store_true")
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="only include these task directory names (default: all tasks)",
    )
    parser.add_argument(
        "--density",
        "--densities",
        dest="densities",
        nargs="+",
        choices=CLUTTER_LEVELS,
        help="only include selected clutter levels, e.g. clean d6 d15",
    )
    parser.add_argument("--list-episodes", action="store_true")
    parser.add_argument("--output", default="contact_attention.gif")
    parser.add_argument("--output-dir", default="contact_attention_gifs")
    parser.add_argument("--cache-subdir", default="beta_weights")
    parser.add_argument(
        "--beta-root",
        default=None,
        help="read caches from <beta-root>/<domain>/<task>/<config>/<cache-subdir>/ "
        "instead of next to the HDF5s; keeps --data-root read-only",
    )
    parser.add_argument("--camera", default="countertop_camera")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--sigma", type=float, default=12.0)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--scale", type=float, default=1.0)
    return parser.parse_args()


def gif_output_path(output_dir: Path, episode: dict) -> Path:
    """Preserve each episode's task/density hierarchy below the output root."""
    label = Path(episode["label"])
    return output_dir / label.parent / f"{label.stem}.gif"


def task_name(episode: dict) -> str:
    parts = Path(episode["label"]).parent.parts
    for index, part in enumerate(parts):
        if part in CLUTTER_LEVELS and index > 0:
            return parts[index - 1]
    return parts[-1] if parts else ""


def filter_episodes(
    episodes: list[dict],
    densities: list[str] | None,
    tasks: list[str] | None,
) -> list[dict]:
    selected_densities = set(densities or ())
    selected_tasks = set(tasks or ())
    filtered = [
        episode
        for episode in episodes
        if (not selected_densities or selected_densities.intersection(Path(episode["label"]).parent.parts))
        and (not selected_tasks or task_name(episode) in selected_tasks)
    ]
    for index, episode in enumerate(filtered):
        episode["id"] = index
    return filtered


def select_episodes(episodes: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.all_episodes:
        return episodes
    if args.episodes is not None:
        episode_numbers = set(args.episodes)
    elif args.episode_range is not None:
        start, end = args.episode_range
        if start > end:
            raise SystemExit("[error] --episode-range START must not exceed END")
        episode_numbers = set(range(start, end + 1))
    else:
        episode_numbers = {0 if args.episode is None else args.episode}

    selected = [
        episode
        for episode in episodes
        if episode_number(Path(episode["label"]).name) in episode_numbers
    ]
    if not selected:
        requested = ", ".join(str(number) for number in sorted(episode_numbers))
        raise SystemExit(f"[error] no matching episode numbers: {requested}")
    return selected


def generate(args, episode: dict, output: Path) -> None:
    cache = load_cache(episode["cache"])
    with h5py.File(episode["path"], "r") as h5:
        num_frames = int(h5[f"observation/{args.camera}/rgb"].shape[0])
        end = num_frames if args.end_frame is None else min(num_frames, args.end_frame)
        indices = range(max(0, args.start_frame), end, args.stride)
        total = len(indices)
        if total == 0:
            raise ValueError(f"{episode['label']}: selected frame range is empty")
        frames = []
        for output_idx, frame in enumerate(indices, start=1):
            image = render_frame(
                h5,
                episode,
                cache,
                frame,
                camera_name=args.camera,
                sigma=args.sigma,
                radius=args.radius,
                patch_size=args.patch_size,
            )
            if args.scale != 1:
                image = image.resize(
                    tuple(max(1, int(value * args.scale)) for value in image.size),
                    Image.Resampling.LANCZOS,
                )
            frames.append(image.quantize(colors=256, method=Image.Quantize.MEDIANCUT))
            if output_idx == 1 or output_idx % 10 == 0 or output_idx == total:
                print(f"\r[render] {output_idx}/{total} source_frame={frame}", end="", flush=True)
    print()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, round(1000 / args.fps)),
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"[done] {episode['label']} -> {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")


def main() -> None:
    args = parse_args()
    if args.stride < 1 or args.fps <= 0 or args.scale <= 0:
        raise SystemExit("[error] stride, fps, and scale must be positive")
    if SIZE % args.patch_size:
        raise SystemExit(f"[error] --patch-size must divide {SIZE}")
    root = str(Path(args.data_root).expanduser())
    episodes = filter_episodes(discover(root, args.cache_subdir, args.beta_root), args.densities, args.tasks)
    if args.list_episodes:
        for episode in episodes:
            print(f"{episode['id']:5d}  {episode['label']}" + ("" if episode["cache"] else " [no cache]"))
        return
    if not episodes:
        density_note = f" for densities {', '.join(args.densities)}" if args.densities else ""
        task_note = f" and tasks {', '.join(args.tasks)}" if args.tasks else ""
        raise SystemExit(f"[error] no episodes under {root}{density_note}{task_note}")
    selected = select_episodes(episodes, args)
    batch_mode = len(selected) > 1 or args.all_episodes or args.episodes is not None or args.episode_range is not None
    if batch_mode:
        output_dir = Path(args.output_dir).expanduser()
        for position, episode in enumerate(selected, start=1):
            print(f"[episode {position}/{len(selected)}; id={episode['id']}] {episode['label']}")
            generate(args, episode, gif_output_path(output_dir, episode))
    else:
        generate(args, selected[0], Path(args.output).expanduser())


if __name__ == "__main__":
    main()
