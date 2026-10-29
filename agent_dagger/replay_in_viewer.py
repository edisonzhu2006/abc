"""Replay a recorded unseen_rollouts.py scene in the interactive MuJoCo viewer.

Rebuilds the scene from the same seed and perturbation, then re-executes the
saved actions. CPU MuJoCo is deterministic, so this reproduces the rollout
without loading the policy. An agent monitor runs alongside (``--monitor
astra``, or the no-API ``stall`` stand-in): when it decides to intervene, an
AGENT INTERVENES banner and a red sphere above the table appear, and playback
pauses until C is pressed. On macOS the viewer must be launched via mjpython,
and mjpython cannot find uv's Python library on its own, so point it there:

    DYLD_LIBRARY_PATH=$HOME/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib \
        uv run mjpython agent_dagger/replay_in_viewer.py --scene bottles_clustered
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import mujoco.viewer
import numpy as np
import tyro

from monitors import MonitorContext, make_monitor
from unseen_rollouts import CAMERAS, SceneName, make_scene

CONTROL_DT = 1 / 30  # one recorded action per 1/30 s of robot time
CHUNK_ACTIONS = 15  # actions per policy chunk in the recording
KEY_CONTINUE = ord("C")
MARKER_POS = (0.6, 0.0, 1.3)  # above the middle of the table, in view of the default camera


@dataclass
class Config:
    scene: SceneName = "baseline"
    rollouts_dir: str = "outputs/unseen_rollouts"
    task: str = "put_plastic_bottles_in_bin"
    seed: int = 0  # must match the seed the rollout was recorded with
    speed: float = 1.0  # playback speed; 2.0 = twice real time
    loop: bool = True
    monitor: Literal["stall", "astra", "claude"] = "stall"  # who decides to intervene; "stall" is the no-API stand-in
    check_every: int = 4  # ask the agent every N chunks (2 s of robot time)
    patience: int = 20  # stall stand-in: chunks without a new bottle in the bin
    pause_on_intervene: bool = True  # freeze on each intervention until C is pressed in the viewer
    prompt: str = "sim throw plastic bottles in bin"


def _texts(status: str, banner: str | None = None, hint: str | None = None):
    rows = [(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, "status", status)]
    if banner:
        rows.append((mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPRIGHT, banner, hint or ""))
    return rows


def _show_marker(viewer, intervene: bool) -> None:
    """Red sphere floating above the table while the agent has called for an intervention."""
    with viewer.lock():
        scn = viewer.user_scn
        scn.ngeom = 0
        if intervene:
            mujoco.mjv_initGeom(scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.06, 0, 0]),
                                np.array(MARKER_POS), np.eye(3).flatten(), np.array([1.0, 0.1, 0.1, 0.85]))
            scn.ngeom = 1


def _evaluate_without_side_effects(env) -> dict:
    """Score the task, then restore the full physics state so the check cannot
    perturb the replay (evaluating touches the solver's warm start)."""
    m, d = env.model, env.data
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    saved = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, saved, spec)
    result = env.evaluate_task()
    mujoco.mj_setState(m, d, saved, spec)
    return result.to_info(squeeze=True) if result else {}


def build_scene(cfg: Config):
    """Reset + perturb + settle exactly as unseen_rollouts.run_scene does."""
    env, _ = make_scene(cfg.task, cfg.scene, cfg.seed, render_cameras=(cfg.monitor in ("astra", "claude")))
    return env


def main(cfg: Config) -> None:
    recording = np.load(Path(cfg.rollouts_dir) / f"{cfg.scene}_sa.npz")
    states, actions = recording["states"], recording["actions"]
    print(f"{cfg.scene}: {len(actions)} actions = {len(actions) * CONTROL_DT:.0f} s of robot time")

    env = build_scene(cfg)
    m, d = env.model, env.data
    # Snapshot the settled start so every loop restarts from the identical state.
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    start = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, start, spec)

    print(f"start-state match: max joint difference {np.abs(env.get_state() - states[0]).max():.2e} rad")
    resume = threading.Event()

    def on_key(keycode: int) -> None:
        if keycode == KEY_CONTINUE:
            resume.set()

    try:
        with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
            while viewer.is_running():
                # No mj_forward here: it would re-run the solver and shift its warm
                # start, and the restored state already carries the right one.
                mujoco.mj_setState(m, d, start, spec)
                monitor = make_monitor(cfg.monitor, cfg.patience)
                _show_marker(viewer, False)
                last = "waiting for first check"
                for i, action in enumerate(actions):
                    if not viewer.is_running():
                        break
                    chunk = i // CHUNK_ACTIONS
                    if i % CHUNK_ACTIONS == 0 and chunk % cfg.check_every == 0 and chunk > 0:
                        result = _evaluate_without_side_effects(env)
                        images = {c: env.get_obs()["images"][c] for c in CAMERAS} if cfg.monitor in ("astra", "claude") else {}
                        decision = monitor(MonitorContext(chunk, images, env.get_state(), cfg.prompt, result))
                        progress = f"{result['num_bottles_in_bin']}/{result['num_active_bottles']} in bin"
                        last = ("INTERVENE - " if decision.intervene else "ok - ") + decision.reason
                        status = f"{cfg.scene} | chunk {chunk} ({chunk * 0.5:.0f} s) | {progress} | agent({monitor.name}): {last}"
                        _show_marker(viewer, decision.intervene)
                        if decision.intervene:
                            print(f"[chunk {chunk:3d}] AGENT INTERVENES: {decision.reason}", flush=True)
                            hint = decision.reason + ("  [press C to continue]" if cfg.pause_on_intervene else "")
                            viewer.set_texts(_texts(status, "AGENT INTERVENES", hint))
                            if cfg.pause_on_intervene:
                                resume.clear()
                                while viewer.is_running() and not resume.is_set():
                                    viewer.sync()
                                    time.sleep(0.05)
                        else:
                            viewer.set_texts(_texts(status))
                    t = time.time()
                    env.step(action, render_obs=False)
                    viewer.sync()
                    time.sleep(max(0.0, CONTROL_DT / cfg.speed - (time.time() - t)))
                else:
                    in_bin = _evaluate_without_side_effects(env)["num_bottles_in_bin"]
                    drift = np.abs(env.get_state() - states[-1]).max()
                    print(f"replay done: bottles in bin {in_bin}, "
                          f"last-step joint difference vs recording {drift:.2e} rad")
                    if not cfg.loop:
                        break
                    time.sleep(1.0)
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
