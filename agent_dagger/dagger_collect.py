"""One DAgger episode: the policy drives, a monitor watches, a corrector takes over, the policy resumes.

The episode lives in a session directory in the same format ``sim_tool.py``
uses, so the corrector works on the live state and the orchestrator picks up
where it left off:

  policy chunk -> policy chunk -> [monitor: intervene] -> freeze session,
  write intervene_request.json, wait ... corrector runs sim_tool commands on
  the session and calls `handback` ... -> reload the handed-back state,
  re-seed the policy's action prefix from it, continue -> ...

Monitors: ``stall`` (sim ground truth, no API) or ``external`` (writes
check_request_<n>.json with the camera image and waits for
check_decision_<n>.json {"intervene": bool, "reason": str} - an agent answers).

    uv run agent_dagger/dagger_collect.py --session outputs/sessions/dagger_s6 --seed 6 --monitor stall
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
import mujoco
import numpy as np
import tyro

from abc_minimal.config import SimEvalConfig
from abc_minimal.policy import DiTInferencePolicy
from monitors import MonitorContext, make_monitor
from sim_tool import GRIPPER_IDX, STATE_SPEC, TASK, _append_actions, _describe, _log, _render, _save_state
from unseen_rollouts import CAMERAS, SceneName, make_scene


@dataclass
class Config:
    session: str
    scene: SceneName = "baseline"
    seed: int = 6
    chunks: int = 120  # 60 s of robot time
    check_every: int = 2  # monitor cadence in chunks (2 = every 1 s of robot time)
    monitor: Literal["stall", "external"] = "stall"
    patience: int = 10  # stall monitor: chunks (0.5 s each) without a new bottle in the bin
    wait_minutes: float = 30.0  # how long to wait for a corrector handback / external decision
    checkpoint: str = "cache/abc_dit_xl_200k_model.pt"
    prompt: str = "sim throw plastic bottles in bin"
    device: str = "mps"
    prefix_length: int = 4
    execute_chunk_dim: int = 15


def _frame(env):
    o = env.get_obs()
    return np.concatenate([o["images"][c].transpose(1, 2, 0) for c in CAMERAS], axis=1)


def _wait_for(path: Path, minutes: float, what: str) -> dict | None:
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        time.sleep(1.0)
    print(f"timed out waiting for {what} ({path.name})", flush=True)
    return None


def main(cfg: Config) -> None:
    session = Path(cfg.session)
    session.mkdir(parents=True, exist_ok=True)
    for old in session.glob("*"):
        old.unlink()
    (session / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print(f"loading {cfg.checkpoint} on {cfg.device}...", flush=True)
    policy = DiTInferencePolicy(Path(cfg.checkpoint), SimEvalConfig(checkpoint=cfg.checkpoint, prompt=cfg.prompt), device=cfg.device)
    monitor = make_monitor(cfg.monitor if cfg.monitor == "stall" else "stall", cfg.patience)  # stall also backs "external" as ground truth
    env, obs = make_scene(TASK, cfg.scene, cfg.seed)
    m, d = env.model, env.data

    meta = {"scene": cfg.scene, "seed": cfg.seed, "step": 0, "history": [], "phase": "policy", "interventions": [],
            "gripper_cmd": {arm: float(obs["state"][i]) for arm, i in GRIPPER_IDX.items()}}
    _save_state(env, session, 0)
    (session / "session.json").write_text(json.dumps(meta))
    video = imageio.get_writer(str(session / "episode.mp4"), fps=6, macro_block_size=1)
    video.append_data(_frame(env))
    phases = []  # (step, phase) for the summary / annotator

    prefix = np.tile(obs["state"], (cfg.prefix_length, 1))
    chunk, n_checks = 0, 0
    try:
        while chunk < cfg.chunks:
            # ---- one policy chunk
            planned = policy.infer({"state": obs["state"], "images": {c: obs["images"][c] for c in CAMERAS}, "prompt": cfg.prompt},
                                   action_prefix=prefix, prefix_length=cfg.prefix_length)
            executed = planned[cfg.prefix_length : cfg.prefix_length + cfg.execute_chunk_dim]
            states = []
            for a in executed:
                states.append(env.get_state().copy())
                env.step(a, render_obs=False)
            _append_actions(session, np.array(states), np.asarray(executed), "policy")
            meta["step"] += len(executed)
            meta["gripper_cmd"] = {arm: float(executed[-1][i]) for arm, i in GRIPPER_IDX.items()}
            prefix = executed[-cfg.prefix_length :]
            obs = env.get_obs()
            video.append_data(_frame(env))
            phases.append((meta["step"], "policy"))
            result = env.evaluate_task().to_info(squeeze=True)
            chunk += 1
            if result["num_bottles_in_bin"] == result["num_active_bottles"]:
                print(f"chunk {chunk}: all bottles in bin", flush=True)
                break
            if chunk % 10 == 0:
                print(f"chunk {chunk}: in_bin={result['num_bottles_in_bin']}/{result['num_active_bottles']}", flush=True)

            # ---- monitor
            if chunk % cfg.check_every:
                continue
            ctx = MonitorContext(chunk=chunk, images={c: obs["images"][c] for c in CAMERAS}, state=obs["state"],
                                 prompt=cfg.prompt, task_eval=result)
            truth = monitor(ctx)  # ground-truth stand-in (also the reference when an external agent decides)
            if cfg.monitor == "external":
                n_checks += 1
                img = _render(env, session, meta["step"])
                (session / f"check_request_{n_checks:03d}.json").write_text(json.dumps(
                    {"chunk": chunk, "robot_time_s": chunk * 0.5, "image": img, "bottles_in_bin": result["num_bottles_in_bin"],
                     "ground_truth_stall": truth.intervene}))
                decision = _wait_for(session / f"check_decision_{n_checks:03d}.json", cfg.wait_minutes, "monitor decision")
                intervene, reason = (bool(decision["intervene"]), decision.get("reason", "")) if decision else (False, "no decision")
            else:
                intervene, reason = truth.intervene, truth.reason
            _log(session, {"cmd": "monitor", "step": meta["step"], "chunk": chunk, "intervene": intervene, "reason": reason,
                           "ground_truth_stall": truth.intervene})
            if not intervene:
                continue

            # ---- hand the live session to the corrector and wait for handback
            print(f"chunk {chunk} ({chunk * 0.5:.0f} s): INTERVENE - {reason}", flush=True)
            _save_state(env, session, meta["step"])
            meta.update(phase="awaiting_corrector", history=[])
            meta.pop("handed_back_at", None)
            (session / "session.json").write_text(json.dumps(meta))
            n_int = len(meta["interventions"]) + 1
            (session / "intervene_request.json").write_text(json.dumps(
                {"intervention": n_int, "chunk": chunk, "step": meta["step"], "reason": reason, "image": _render(env, session, meta["step"]),
                 "scene": _describe(env, meta)}, indent=1))
            t0 = time.time()
            handed = None
            deadline = time.time() + cfg.wait_minutes * 60
            while time.time() < deadline:
                try:
                    cur = json.loads((session / "session.json").read_text())
                except json.JSONDecodeError:
                    cur = {}
                if cur.get("handed_back_at") is not None:
                    handed = cur
                    break
                time.sleep(1.0)
            if handed is None:
                print("corrector never handed back; stopping the episode", flush=True)
                break
            # ---- resume the policy from the handed-back state
            meta = handed
            snap = np.load(session / f"state_{meta['step']:04d}.npz")["state"]
            mujoco.mj_setState(m, d, snap, STATE_SPEC)
            mujoco.mj_forward(m, d)
            obs = env.get_obs()
            prefix = np.tile(obs["state"], (cfg.prefix_length, 1))
            meta["phase"] = "policy"
            meta["interventions"].append({"n": n_int, "flag_chunk": chunk, "start_step": (session / "intervene_request.json").exists() and json.loads((session / "intervene_request.json").read_text())["step"],
                                          "end_step": meta["step"], "wall_s": round(time.time() - t0)})
            meta.pop("handed_back_at")
            (session / "intervene_request.json").unlink(missing_ok=True)
            (session / "session.json").write_text(json.dumps(meta))
            phases.append((meta["step"], "agent"))
            video.append_data(_frame(env))
            result = env.evaluate_task().to_info(squeeze=True)
            print(f"handback after {meta['interventions'][-1]['end_step'] - meta['interventions'][-1]['start_step']} agent steps; "
                  f"in_bin={result['num_bottles_in_bin']}/{result['num_active_bottles']}; policy resumes", flush=True)
            if hasattr(monitor, "_last_progress_chunk"):
                monitor._last_progress_chunk = chunk
                monitor._best = result["num_bottles_in_bin"]
    finally:
        video.close()
        result = env.evaluate_task().to_info(squeeze=True)
        summary = {"final_in_bin": result["num_bottles_in_bin"], "active": result["num_active_bottles"],
                   "success": bool(result["ever_success"]), "chunks": chunk, "steps": meta["step"],
                   "interventions": meta["interventions"], "phases": phases}
        (session / "summary.json").write_text(json.dumps(summary, indent=1))
        meta["phase"] = "done"
        (session / "session.json").write_text(json.dumps(meta))
        env.close()
        print("episode done:", json.dumps({k: v for k, v in summary.items() if k != "phases"}), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
