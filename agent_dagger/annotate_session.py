"""Render a corrector session as an annotated replay video.

Re-simulates the episode from scratch (the policy's recorded chunks, then the
agent's recorded actions), renders the three cameras, and overlays per frame:
the phase (POLICY vs AGENT INTERVENTION, red border while the agent drives),
the agent's current EEF command and its note, the live grasp-point position of
the commanded arm, and the ground-truth bottle count.

    uv run agent_dagger/annotate_session.py --session outputs/sessions/agent_s6_c28
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

from sim_tool import GRASP_OFFSET, GRIPPER_IDX, _bottle_geometry, _site
from unseen_rollouts import CAMERAS, make_scene

PANEL_H = 150


@dataclass
class Config:
    session: str
    out: str | None = None  # default: <session>/replay_annotated.mp4
    every: int = 5  # render one frame every N sim steps (5 = 6 fps of robot time)
    fps: int = 12  # playback frames per second (12 with every=5 -> 2x real time)
    scale: int = 2


def _surviving_commands(session: Path, agent_actions: np.ndarray, start_step: int) -> tuple[list[dict], np.ndarray]:
    """Rebuild the executed trajectory from log.jsonl.

    actions.npz holds every command's actions in order, including ones later
    undone, so each action-producing log entry owns the next block of
    ``end - start`` rows. An ``undo`` discards the previous surviving block.
    Returns the surviving commands (with their step spans renumbered along the
    executed trajectory) and the concatenated surviving actions."""
    entries = [json.loads(l) for l in (session / "log.jsonl").read_text().splitlines()]
    blocks, offset, prev_step, handback = [], 0, start_step, None
    for e in entries:
        if e["cmd"] == "undo":
            if blocks:
                blocks.pop()
            prev_step = e["step"]
            continue
        if e["cmd"] == "handback":
            handback = e
            continue
        n = e["step"] - prev_step
        blocks.append({"entry": e, "rows": agent_actions[offset : offset + n]})
        offset += n
        prev_step = e["step"]
    spans, step, rows = [], start_step, []
    for b in blocks:
        e, a = b["entry"], b["entry"].get("args", {})
        if e["cmd"] == "move":
            text = (f"move {a['arm']} -> ({a['to'][0]:.2f}, {a['to'][1]:.2f}, {a['to'][2]:.2f})"
                    f"  pointing={a.get('pointing', 'keep')}" + (f"  across={a['across']}" if a.get("across") else "")
                    + (f"  gripper={'closed' if a['gripper'] == 0 else 'open' if a['gripper'] == 1 else a['gripper']}" if a.get("gripper") is not None else ""))
            arm = a["arm"]
        elif e["cmd"] == "hold":
            text, arm = f"hold {a.get('steps', '')} steps", None
        else:
            text, arm = f"{e['cmd']}", None
        spans.append({"start": step, "end": step + len(b["rows"]), "text": text, "note": a.get("note", ""), "arm": arm})
        step += len(b["rows"]); rows.append(b["rows"])
    if handback is not None:
        spans.append({"start": step, "end": step, "text": "HANDBACK: " + handback.get("note", ""), "note": "", "arm": None})
    return spans, (np.concatenate(rows) if rows else np.zeros((0, 14), np.float32)), (step if handback is not None else None)


def main(cfg: Config) -> None:
    session = Path(cfg.session)
    meta = json.loads((session / "session.json").read_text())
    rec = np.load(meta["replayed_from"]) if meta.get("replayed_from") else None
    n_policy = int(meta["history"][0]) if meta.get("history") else 0
    agent = np.load(session / "actions.npz")
    spans, agent_rows, handback_step = _surviving_commands(session, agent["actions"], n_policy)
    actions = np.concatenate([rec["actions"][:n_policy], agent_rows]) if rec is not None else agent_rows

    env, obs = make_scene("put_plastic_bottles_in_bin", meta["scene"], meta["seed"])
    m, d = env.model, env.data
    out = Path(cfg.out) if cfg.out else session / "replay_annotated.mp4"
    writer = imageio.get_writer(str(out), fps=cfg.fps, macro_block_size=1)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)

    def frame(step: int, action: np.ndarray):
        o = env.get_obs()
        cams = np.concatenate([o["images"][c].transpose(1, 2, 0) for c in CAMERAS], axis=1)
        img = Image.fromarray(cams).resize((cams.shape[1] * cfg.scale, cams.shape[0] * cfg.scale), Image.NEAREST)
        canvas = Image.new("RGB", (img.width, img.height + PANEL_H), "black")
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        agent_phase = step >= n_policy and (handback_step is None or step < handback_step)
        span = next((s for s in spans if s["start"] <= step < s["end"]), None) if agent_phase else None
        result = env.evaluate_task().to_info(squeeze=True)
        y = img.height + 6
        phase = "AGENT INTERVENTION (EEF commands -> IK -> joints)" if agent_phase else ("POLICY (resumed after handback)" if step >= n_policy else "POLICY")
        draw.text((8, y), f"t = {step / 30:5.1f} s   {phase}   |   sim truth: {result['num_bottles_in_bin']}/{result['num_active_bottles']} in bin",
                  fill=(255, 90, 90) if agent_phase else (120, 230, 120), font=font)
        y += 26
        if span is not None:
            draw.text((8, y), f"command: {span['text']}", fill="white", font=font); y += 24
            if span["arm"]:
                sid = _site(env, span["arm"]); R = d.site_xmat[sid].reshape(3, 3)
                gp = d.site_xpos[sid] + GRASP_OFFSET * R[:, 2]
                gap = float(env.get_state()[GRIPPER_IDX[span["arm"]]]) * 0.095
                draw.text((8, y), f"{span['arm']} EEF now: grasp point ({gp[0]:.3f}, {gp[1]:.3f}, {gp[2]:.3f})  jaw gap {gap:.3f} m  "
                                  f"joint targets this step: {np.round(action[0:7] if span['arm'] == 'left' else action[7:14], 2).tolist()}",
                          fill=(200, 200, 255), font=small); y += 22
            if span["note"]:
                for line in textwrap.wrap("why: " + span["note"], width=int(img.width / 9)):
                    draw.text((8, y), line, fill=(255, 230, 150), font=small); y += 19
        elif agent_phase:
            draw.text((8, y), "agent: (between commands)", fill="gray", font=small)
        if agent_phase:
            draw.rectangle([0, 0, img.width - 1, img.height - 1], outline=(255, 60, 60), width=5)
        writer.append_data(np.asarray(canvas))

    for step, action in enumerate(actions):
        if step % cfg.every == 0:
            frame(step, action)
        env.step(action, render_obs=False)
    frame(len(actions), actions[-1])
    writer.close(); env.close()
    print(f"wrote {out}: {len(actions)} steps ({n_policy} policy + {len(actions) - n_policy} agent), one frame per {cfg.every} steps")


if __name__ == "__main__":
    main(tyro.cli(Config))
