"""Overlay an agent's intervention labels (and the sim's ground truth) onto a rollout video.

Reads the rollout mp4 (one frame per chunk), the labels.json a monitor agent
wrote for it, and the ground-truth progress series from the matching _sa.npz,
and writes <video stem>_annotated.mp4 with a text panel under the cameras:

    uv run agent_dagger/annotate_video.py --video outputs/failure_sweep/baseline_s3.mp4 \
        --labels outputs/failure_sweep/baseline_s3_sheets/labels.json
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

CHUNK_S = 0.5
PANEL_H = 120


@dataclass
class Config:
    video: str
    labels: str
    npz: str | None = None  # default: <video stem>_sa.npz next to the video
    out: str | None = None  # default: <video stem>_annotated.mp4
    fps: int = 4  # playback speed in chunks per second (2 = real time)
    hold_flag_chunks: int = 8  # keep an INTERVENE banner up this many chunks after it fires
    scale: int = 2


def main(cfg: Config) -> None:
    video = Path(cfg.video)
    frames = imageio.mimread(str(video), memtest=False)
    labels = json.load(open(cfg.labels))
    by_chunk = {int(r["chunk"]): r for r in labels}
    npz_path = Path(cfg.npz) if cfg.npz else video.with_name(video.stem + "_sa.npz")
    gt = np.load(npz_path)["in_bin_per_chunk"] if npz_path.exists() else None

    out_path = Path(cfg.out) if cfg.out else video.with_name(video.stem + "_annotated.mp4")
    writer = imageio.get_writer(str(out_path), fps=cfg.fps, macro_block_size=1)
    last_label, last_flag_chunk = None, -10_000
    for chunk, frame in enumerate(frames):
        if chunk in by_chunk:
            last_label = by_chunk[chunk]
            if last_label["intervene"]:
                last_flag_chunk = chunk
        cams = Image.fromarray(frame).resize((frame.shape[1] * cfg.scale, frame.shape[0] * cfg.scale), Image.NEAREST)
        canvas = Image.new("RGB", (cams.width, cams.height + PANEL_H), "black")
        canvas.paste(cams, (0, 0))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=18)
        y = cams.height + 6
        gt_text = f"   sim truth: {int(gt[chunk - 1]) if gt is not None and chunk > 0 else 0} in bin" if gt is not None else ""
        draw.text((8, y), f"chunk {chunk:3d}   t = {chunk * CHUNK_S:5.1f} s{gt_text}", fill="white", font=font)
        y += 24
        if last_label is None:
            draw.text((8, y), "agent: (no checkpoint yet)", fill="gray", font=font)
        else:
            flagged = chunk - last_flag_chunk < cfg.hold_flag_chunks
            head = (f"agent @ {last_label['t']:.0f}s: INTERVENE  [{last_label['failure_mode']}]" if flagged
                    else f"agent @ {last_label['t']:.0f}s: ok  (sees {last_label['bottles_in_bin']} in bin)")
            draw.text((8, y), head, fill=(255, 80, 80) if flagged else (120, 230, 120), font=font)
            y += 24
            for line in textwrap.wrap(last_label["reason"], width=int(cams.width / 10)):
                draw.text((8, y), line, fill="white", font=font)
                y += 22
        if chunk - last_flag_chunk < cfg.hold_flag_chunks:
            draw.rectangle([0, 0, cams.width - 1, cams.height - 1], outline=(255, 60, 60), width=4)
        writer.append_data(np.asarray(canvas))
    writer.close()
    print(f"wrote {out_path} ({len(frames)} frames at {cfg.fps} chunks/s)")


if __name__ == "__main__":
    main(tyro.cli(Config))
