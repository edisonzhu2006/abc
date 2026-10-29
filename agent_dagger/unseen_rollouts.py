"""Roll out an ABC policy on the bottles scene, plain or pushed into unseen states.

Each scene edits the live MuJoCo state right after reset (move/tip the bin,
cluster the bottles, change the lighting), lets it settle, then runs the policy
for a fixed number of 15-action chunks. Per scene it writes a video, the
(state, action) pairs, and a line in summary.json.

    uv run agent_dagger/unseen_rollouts.py                          # all scenes
    uv run agent_dagger/unseen_rollouts.py --scenes bottles_clustered --chunks 120
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import imageio
import mujoco
import numpy as np
import tyro

import abc_sim
from abc_minimal.config import SimEvalConfig
from abc_minimal.policy import DiTInferencePolicy

CAMERAS = ("top", "left", "right")


# Scene perturbations: each one edits env.model / env.data after reset.

def _joint(m: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)


def _position(env, name: str) -> np.ndarray:
    a = env.model.jnt_qposadr[_joint(env.model, name)]
    return env.data.qpos[a : a + 3].copy()


def _place(env, name: str, pos=None, rot=None) -> None:
    """Move a free object. ``rot`` is applied on top of its current orientation:
    the bin's upright pose is already a 90 degree roll, so overwriting it tips it."""
    m, d = env.model, env.data
    a = m.jnt_qposadr[_joint(m, name)]
    if pos is not None:
        d.qpos[a : a + 3] = pos
    if rot is not None:
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.asarray(rot, float), d.qpos[a + 3 : a + 7].copy())
        d.qpos[a + 3 : a + 7] = q
    v = m.jnt_dofadr[_joint(m, name)]
    d.qvel[v : v + 6] = 0


def _axis_quat(axis, degrees: float) -> np.ndarray:
    half = np.radians(degrees) / 2
    return np.r_[np.cos(half), np.sin(half) * np.asarray(axis, float)]


def baseline(env) -> None:
    pass


def bin_tipped(env) -> None:
    """Bin knocked onto its side: unplaceable until something stands it back up."""
    p = _position(env, "bin_joint")
    _place(env, "bin_joint", [p[0], p[1], 0.88], _axis_quat([0, 1, 0], 90))


def bottles_clustered(env) -> None:
    """All bottles piled in one spot on the table.

    Bottles are ~19 cm long, so placing them a few cm apart makes them start
    inside each other and the solver flings them apart. Instead drop them one
    at a time from just above the pile and let each settle before the next."""
    hold = env.get_state().copy()
    rng = np.random.default_rng(0)
    centre = np.array([0.60, 0.30])
    for i in range(1, 7):
        p = _position(env, f"bottle_{i}_joint")
        xy = centre + rng.uniform(-0.04, 0.04, size=2)
        _place(env, f"bottle_{i}_joint", [xy[0], xy[1], p[2] + 0.06 + 0.05 * i])
        mujoco.mj_forward(env.model, env.data)
        for _ in range(20):  # ~0.7 s: let it land on the pile
            env.step(hold, render_obs=False)


def visual_shift(env) -> None:
    """Dim orange key light and dark ambient: same physics, unfamiliar images."""
    env.model.light_diffuse[:] = [0.9, 0.5, 0.3]
    env.model.light_ambient[:] = [0.05, 0.05, 0.1]


def bin_far_right(env) -> None:
    """Bin beyond its training range (x 0.566-0.646, |y| <= 0.2), turned 90 degrees."""
    z = _position(env, "bin_joint")[2]
    _place(env, "bin_joint", [0.70, -0.32, z], _axis_quat([0, 0, 1], 90))
    for i in range(1, 7):
        p = _position(env, f"bottle_{i}_joint")
        if np.hypot(p[0] - 0.70, p[1] + 0.32) < 0.2:
            _place(env, f"bottle_{i}_joint", [0.45, 0.08 * i - 0.25, p[2]])


SceneName = Literal["baseline", "bin_tipped", "bottles_clustered", "visual_shift", "bin_far_right"]
SCENES: dict[str, Callable] = {
    f.__name__: f for f in (baseline, bin_tipped, bottles_clustered, visual_shift, bin_far_right)
}


@dataclass
class Config:
    scenes: tuple[SceneName, ...] = tuple(SCENES)
    checkpoint: str = "cache/abc_dit_xl_200k_model.pt"
    prompt: str = "sim throw plastic bottles in bin"  # the 200k parent's wording for this scene
    task: str = "put_plastic_bottles_in_bin"  # abc_sim task (scene + evaluator)
    seed: int = 0  # scene randomization seed; same seed = same starting layout
    seeds: tuple[int, ...] = ()  # run each scene on several seeds instead (outputs named <scene>_s<seed>)
    chunks: int = 60  # x 15 actions at 30 Hz = 0.5 s of robot time per chunk
    execute_chunk_dim: int = 15
    prefix_length: int = 4
    device: str = "mps"  # "cuda" on an NVIDIA box, "cpu" as a last resort
    output_dir: str = "outputs/unseen_rollouts"


def make_scene(task: str, scene: str, seed: int, render_cameras: bool = True):
    """Reset, apply the scene's perturbation, and let it settle. Returns (env, obs).

    Shared by the rollout, replay, and live-monitor scripts so all three start
    from the identical state for a given (scene, seed)."""
    env = abc_sim.make_env(task=task, render_cameras=render_cameras, camera_backend="mujoco",
                           camera_height=168, camera_width=224)
    obs, _ = env.reset(seed=seed, randomize=True)
    SCENES[scene](env)
    mujoco.mj_forward(env.model, env.data)
    for _ in range(30):  # ~1 s with the arms held still so edited objects settle
        env.step(obs["state"], render_obs=False)
    return env, (env.get_obs() if render_cameras else None)


def _frame(obs) -> np.ndarray:
    return np.concatenate([obs["images"][c].transpose(1, 2, 0) for c in CAMERAS], axis=1)


def _bottle_heights(env) -> np.ndarray:
    """Height of each active bottle's origin above the table; a lift that ends
    outside the bin shows up as a rise and fall in this series."""
    m, d = env.model, env.data
    heights = []
    for i in range(1, 7):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"bottle_{i}_joint")
        if jid >= 0:
            heights.append(d.qpos[m.jnt_qposadr[jid] + 2])
    return np.array(heights)


def run_scene(name: str, policy: DiTInferencePolicy, cfg: Config, out: Path, seed: int,
              label: str) -> dict:
    t0 = time.time()
    env, obs = make_scene(cfg.task, name, seed)
    try:
        video = imageio.get_writer(str(out / f"{label}.mp4"), fps=6, macro_block_size=1)
        video.append_data(_frame(obs))
        states, actions, chunk_ids = [], [], []
        in_bin_per_chunk, max_height_per_chunk = [], []
        # First chunk: condition on the current pose, as eval_policy.py does.
        prefix = np.tile(obs["state"], (cfg.prefix_length, 1))
        best = 0
        for chunk in range(cfg.chunks):
            policy_obs = {"state": obs["state"],
                          "images": {c: obs["images"][c] for c in CAMERAS},
                          "prompt": cfg.prompt}
            planned = policy.infer(policy_obs, action_prefix=prefix, prefix_length=cfg.prefix_length)
            executed = planned[cfg.prefix_length : cfg.prefix_length + cfg.execute_chunk_dim]
            for action in executed:
                states.append(env.get_state().copy())
                actions.append(action.copy())
                chunk_ids.append(chunk)
                env.step(action, render_obs=False)
            prefix = executed[-cfg.prefix_length :]
            obs = env.get_obs()
            video.append_data(_frame(obs))
            result = env.evaluate_task().to_info(squeeze=True)
            best = max(best, result["num_bottles_in_bin"])
            in_bin_per_chunk.append(result["num_bottles_in_bin"])
            max_height_per_chunk.append(float(_bottle_heights(env).max()))
            if chunk % 10 == 9:
                print(f"  {label} chunk {chunk + 1}/{cfg.chunks}: "
                      f"in_bin={result['num_bottles_in_bin']}/{result['num_active_bottles']}", flush=True)
            if result["num_bottles_in_bin"] == result["num_active_bottles"]:
                print(f"  {label}: all bottles in bin at chunk {chunk + 1}", flush=True)
                break
        video.close()
    finally:
        env.close()

    np.savez(out / f"{label}_sa.npz", states=np.array(states, np.float32),
             actions=np.array(actions, np.float32), chunk=np.array(chunk_ids),
             in_bin_per_chunk=np.array(in_bin_per_chunk), max_bottle_height_per_chunk=np.array(max_height_per_chunk))
    return {
        "scene": name,
        "seed": seed,
        "final_in_bin": result["num_bottles_in_bin"],
        "best_in_bin": best,
        "active": result["num_active_bottles"],
        "success": bool(result["ever_success"]),
        "chunks_run": len(in_bin_per_chunk),
        "pairs": len(states),
        "wall_s": round(time.time() - t0),
    }


def main(cfg: Config) -> None:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"loading {cfg.checkpoint} on {cfg.device} (20-100 s, prints nothing until done)...", flush=True)
    policy = DiTInferencePolicy(Path(cfg.checkpoint), SimEvalConfig(checkpoint=cfg.checkpoint, prompt=cfg.prompt),
                                device=cfg.device)
    print(f"policy loaded in {time.time() - t0:.0f}s", flush=True)

    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    seeds = cfg.seeds or (cfg.seed,)
    for name in cfg.scenes:
        for seed in seeds:
            label = f"{name}_s{seed}" if cfg.seeds else name
            summary[label] = run_scene(name, policy, cfg, out, seed, label)
            print(f"{label}: {summary[label]}", flush=True)
            summary_path.write_text(json.dumps(summary, indent=2))
    print(f"done -> {out}/", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
