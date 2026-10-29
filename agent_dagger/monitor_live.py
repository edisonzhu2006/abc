"""Watch the ABC policy live in the MuJoCo viewer while a monitor decides when to intervene.

Every ``--check-every`` chunks the monitor sees the three camera images (and,
for the stall monitor, the sim's ground-truth progress). When it says
intervene, the viewer shows its reason, the sim pauses until you press C, and
the decision plus the frames it saw are logged. Nothing takes over the arms
yet: this is the "when" half of DAgger.

    DYLD_LIBRARY_PATH=$HOME/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib \
        uv run mjpython agent_dagger/monitor_live.py --scene bottles_clustered
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import imageio
import mujoco
import mujoco.viewer
import numpy as np
import tyro

from abc_minimal.config import SimEvalConfig
from abc_minimal.policy import DiTInferencePolicy
from monitors import MonitorContext, make_monitor
from unseen_rollouts import CAMERAS, SceneName, make_scene

CONTROL_DT = 1 / 30
KEY_CONTINUE = ord("C")


@dataclass
class Config:
    scene: SceneName = "bottles_clustered"
    seed: int = 0
    chunks: int = 240  # 120 s of robot time
    check_every: int = 4  # ask the monitor every N chunks (2 s of robot time)
    monitor: Literal["stall", "astra", "claude"] = "stall"
    patience: int = 20  # stall monitor: chunks without a new bottle in the bin
    pause_on_intervene: bool = True  # freeze until C is pressed in the viewer
    realtime: bool = True  # play each chunk's actions at 30 Hz so motion is watchable
    checkpoint: str = "cache/abc_dit_xl_200k_model.pt"
    prompt: str = "sim throw plastic bottles in bin"
    task: str = "put_plastic_bottles_in_bin"
    device: str = "mps"
    prefix_length: int = 4
    execute_chunk_dim: int = 15
    output_dir: str = "outputs/monitor_live"


def _texts(status: str, banner: str | None = None, hint: str | None = None):
    rows = [(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, "status", status)]
    if banner:
        rows.append((mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPRIGHT, banner, hint or ""))
    return rows


def main(cfg: Config) -> None:
    run_dir = Path(cfg.output_dir) / f"{cfg.scene}_{cfg.monitor}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print(f"loading {cfg.checkpoint} on {cfg.device} (20-100 s)...", flush=True)
    policy = DiTInferencePolicy(Path(cfg.checkpoint), SimEvalConfig(checkpoint=cfg.checkpoint, prompt=cfg.prompt),
                                device=cfg.device)
    monitor = make_monitor(cfg.monitor, cfg.patience)
    env, obs = make_scene(cfg.task, cfg.scene, cfg.seed)
    print(f"scene {cfg.scene} ready; monitor={monitor.name}, checking every {cfg.check_every} chunks", flush=True)

    resume = threading.Event()

    def on_key(keycode: int) -> None:
        if keycode == KEY_CONTINUE:
            resume.set()

    log = open(run_dir / "decisions.jsonl", "w")
    interventions = 0
    try:
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
            prefix = np.tile(obs["state"], (cfg.prefix_length, 1))
            last_reason = "waiting for first check"
            for chunk in range(cfg.chunks):
                if not viewer.is_running():
                    break
                result = env.evaluate_task().to_info(squeeze=True)
                progress = f"{result['num_bottles_in_bin']}/{result['num_active_bottles']} in bin"
                status = f"{cfg.scene} | chunk {chunk} ({chunk * 0.5:.0f} s) | {progress} | {monitor.name}: {last_reason}"

                if chunk % cfg.check_every == 0 and chunk > 0:
                    viewer.set_texts(_texts(status, "monitor checking..."))
                    viewer.sync()
                    ctx = MonitorContext(chunk=chunk, images={c: obs["images"][c] for c in CAMERAS},
                                         state=obs["state"], prompt=cfg.prompt, task_eval=result)
                    t = time.time()
                    decision = monitor(ctx)
                    entry = {"chunk": chunk, "robot_time_s": chunk * 0.5, "intervene": decision.intervene,
                             "reason": decision.reason, "monitor_s": round(time.time() - t, 2),
                             "num_bottles_in_bin": result["num_bottles_in_bin"], "raw": decision.raw}
                    log.write(json.dumps(entry) + "\n")
                    log.flush()
                    last_reason = ("INTERVENE - " if decision.intervene else "ok - ") + decision.reason
                    print(f"[chunk {chunk:3d}] {last_reason}", flush=True)
                    if decision.intervene:
                        interventions += 1
                        frame = np.concatenate([obs["images"][c].transpose(1, 2, 0) for c in CAMERAS], axis=1)
                        imageio.imwrite(run_dir / f"intervene_chunk{chunk:03d}.png", frame)
                        if cfg.pause_on_intervene:
                            resume.clear()
                            viewer.set_texts(_texts(status, "INTERVENE", f"{decision.reason}  [press C to continue]"))
                            while viewer.is_running() and not resume.is_set():
                                viewer.sync()
                                time.sleep(0.05)
                    status = f"{cfg.scene} | chunk {chunk} ({chunk * 0.5:.0f} s) | {progress} | {monitor.name}: {last_reason}"

                if result["num_bottles_in_bin"] == result["num_active_bottles"]:
                    viewer.set_texts(_texts(status, "DONE", "all bottles in bin - close the window"))
                    break

                viewer.set_texts(_texts(status, "policy planning..."))
                viewer.sync()
                planned = policy.infer({"state": obs["state"], "images": {c: obs["images"][c] for c in CAMERAS},
                                        "prompt": cfg.prompt},
                                       action_prefix=prefix, prefix_length=cfg.prefix_length)
                executed = planned[cfg.prefix_length : cfg.prefix_length + cfg.execute_chunk_dim]
                viewer.set_texts(_texts(status))
                for action in executed:
                    t = time.time()
                    env.step(action, render_obs=False)
                    viewer.sync()
                    if cfg.realtime:
                        time.sleep(max(0.0, CONTROL_DT - (time.time() - t)))
                prefix = executed[-cfg.prefix_length :]
                obs = env.get_obs()
            else:
                viewer.set_texts(_texts(status, "OUT OF TIME", f"{cfg.chunks} chunks - close the window"))

            print(f"finished: {interventions} intervention(s); log in {run_dir}/", flush=True)
            while viewer.is_running():  # keep the final scene up until the window is closed
                viewer.sync()
                time.sleep(0.1)
    finally:
        log.close()
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
