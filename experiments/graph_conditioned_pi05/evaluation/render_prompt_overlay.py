#!/usr/bin/env python3
"""Render an episode's video with a side panel showing the live VLA prompt.

Reads ``episode0.mp4`` and ``episode0_action_trace.npz`` from an episode
directory and writes a new video with the original rollout on the left and a
per-frame text panel on the right showing the prompt the policy received at
that frame (plus frame index, phase, and substage for context), so the
prompt can be watched updating in sync with the rollout instead of read out
of the trace separately.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
VIDEO_SCALE = 2
PANEL_WIDTH = 560
PANEL_BACKGROUND = (24, 24, 26)
PANEL_TEXT = (235, 235, 235)
PANEL_MUTED = (150, 150, 155)
PANEL_ACCENT = (110, 170, 255)
MARGIN = 20
LINE_SPACING = 8


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(DEFAULT_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap one line of text to fit max_width, preserving existing breaks."""
    lines = []
    for raw_line in text.split("\n"):
        if not raw_line:
            lines.append("")
            continue
        # textwrap needs a rough char width; refine by measuring actual pixel width.
        approx_chars = max(10, int(max_width / max(1, font.getlength("m"))))
        for wrapped in textwrap.wrap(raw_line, width=approx_chars) or [""]:
            lines.append(wrapped)
    return lines


def _draw_text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    max_width: int,
) -> int:
    """Draw word-wrapped text, return the y-coordinate just below it."""
    x, y = xy
    for line in _wrap(text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + LINE_SPACING
    return y


def _render_panel(
    height: int,
    frame_index: int,
    row: dict,
    label_font: ImageFont.FreeTypeFont,
    prompt_font: ImageFont.FreeTypeFont,
) -> np.ndarray:
    panel = Image.new("RGB", (PANEL_WIDTH, height), PANEL_BACKGROUND)
    draw = ImageDraw.Draw(panel)
    max_text_width = PANEL_WIDTH - 2 * MARGIN
    y = MARGIN

    header = f"frame {frame_index}"
    phase = row.get("phase")
    if phase:
        header += f"   phase={phase}"
    y = _draw_text_block(draw, (MARGIN, y), header, label_font, PANEL_ACCENT, max_text_width)

    substage = None
    action_intent = row.get("action_intent")
    if action_intent:
        try:
            substage = json.loads(action_intent).get("grasp_substage")
        except (TypeError, ValueError):
            substage = None
    held_arm = row.get("held_arm") or None
    detail_bits = []
    if substage:
        detail_bits.append(f"substage={substage}")
    if held_arm:
        detail_bits.append(f"held_arm={held_arm}")
    if detail_bits:
        y = _draw_text_block(
            draw, (MARGIN, y), "   ".join(detail_bits), label_font, PANEL_MUTED, max_text_width
        )

    y += MARGIN
    draw.line((MARGIN, y, PANEL_WIDTH - MARGIN, y), fill=(70, 70, 75), width=1)
    y += MARGIN

    prompt = str(row.get("prompt") or "")
    _draw_text_block(draw, (MARGIN, y), prompt, prompt_font, PANEL_TEXT, max_text_width)

    return np.asarray(panel)


def render(episode_dir: Path, output_path: Path, font_size: int) -> None:
    video_path = episode_dir / "episode0.mp4"
    trace_path = episode_dir / "episode0_action_trace.npz"
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)

    data = np.load(trace_path, allow_pickle=True)
    n_rows = len(data["frame"]) if "frame" in data.files else len(data["prompt"])
    rows = [
        {key: data[key][i] for key in data.files if len(data[key]) == n_rows}
        for i in range(n_rows)
    ]

    meta = iio.immeta(video_path)
    fps = float(meta.get("fps", 10.0))

    label_font = _font(font_size - 2)
    prompt_font = _font(font_size)

    writer_frames = []
    video_height = None
    for index, frame in enumerate(iio.imiter(video_path, plugin="pyav")):
        if index >= n_rows:
            break
        image = Image.fromarray(frame)
        scaled = image.resize(
            (image.width * VIDEO_SCALE, image.height * VIDEO_SCALE), Image.NEAREST
        )
        video_height = scaled.height
        panel = _render_panel(scaled.height, index, rows[index], label_font, prompt_font)
        combined = np.hstack([np.asarray(scaled), panel])
        writer_frames.append(combined)

    if not writer_frames:
        raise RuntimeError("no frames rendered -- check video/trace lengths")
    if len(writer_frames) < n_rows:
        print(
            f"warning: video had {len(writer_frames)} frames, trace had {n_rows} rows; "
            f"trailing trace rows were dropped"
        )

    iio.imwrite(output_path, np.stack(writer_frames), fps=fps, codec="libx264")
    print(f"wrote {output_path} ({len(writer_frames)} frames @ {fps:.1f}fps)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode_dir", type=Path,
        help="episode directory containing episode0.mp4 and episode0_action_trace.npz",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="output video path (default: episode0_with_prompt.mp4 inside episode_dir)",
    )
    parser.add_argument(
        "--font-size", type=int, default=20,
        help="prompt panel font size in points (default: 20)",
    )
    args = parser.parse_args()

    episode_dir = args.episode_dir.expanduser().resolve()
    output_path = args.output or (episode_dir / "episode0_with_prompt.mp4")
    render(episode_dir, output_path, args.font_size)


if __name__ == "__main__":
    main()
