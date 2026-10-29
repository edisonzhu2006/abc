"""Turn a rollout video into labelled contact sheets a vision model can review.

Each row is one checkpoint: the overhead camera, then the left and right wrist
cameras, stamped with the chunk index and robot time. Rows are sampled every
``--every`` chunks (the same cadence the live monitor checks at) and stacked
``--rows`` per sheet, written in order as sheet_00.png, sheet_01.png, ...

    uv run agent_dagger/video_sheets.py --video outputs/failure_sweep/baseline_s3.mp4
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import tyro
from PIL import Image, ImageDraw

CAM_W, CAM_H = 224, 168  # per-camera tile size in the rollout videos
LABEL_H = 22
CHUNK_S = 0.5  # robot time per chunk


@dataclass
class Config:
    video: str
    out_dir: str | None = None  # default: <video dir>/<video stem>_sheets/
    every: int = 4  # sample one row every N chunks
    rows: int = 6  # rows per sheet
    scale: int = 2  # upscale factor so the tiles are legible


def main(cfg: Config) -> None:
    video = Path(cfg.video)
    out = Path(cfg.out_dir) if cfg.out_dir else video.parent / f"{video.stem}_sheets"
    out.mkdir(parents=True, exist_ok=True)
    frames = imageio.mimread(str(video), memtest=False)  # frame i = state after chunk i (frame 0 = start)
    picks = list(range(0, len(frames), cfg.every))
    if picks[-1] != len(frames) - 1:
        picks.append(len(frames) - 1)

    rows = []
    for chunk in picks:
        tile = Image.fromarray(frames[chunk]).resize((3 * CAM_W * cfg.scale, CAM_H * cfg.scale), Image.NEAREST)
        row = Image.new("RGB", (tile.width, tile.height + LABEL_H * cfg.scale), "black")
        draw = ImageDraw.Draw(row)
        draw.text((6, 4), f"chunk {chunk:3d}   t = {chunk * CHUNK_S:5.1f} s      "
                          f"[overhead | left wrist | right wrist]", fill="white")
        row.paste(tile, (0, LABEL_H * cfg.scale))
        rows.append(np.asarray(row))

    sheets = [rows[i : i + cfg.rows] for i in range(0, len(rows), cfg.rows)]
    for n, group in enumerate(sheets):
        imageio.imwrite(out / f"sheet_{n:02d}.png", np.concatenate(group, axis=0))
    print(f"{video.name}: {len(frames)} chunks -> {len(rows)} checkpoints on {len(sheets)} sheets in {out}/")


if __name__ == "__main__":
    main(tyro.cli(Config))
