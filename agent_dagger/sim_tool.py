"""Command-line sim session for a corrector agent: observe the scene, output actions, repeat.

A session is a directory holding the scene identity (scene, seed), the full
physics state after each command, every action applied (the (s, a) pairs for
DAgger), and the camera renders. Every command is a fresh process: it rebuilds
the scene, restores the saved state, does its work, and saves again.

    T=agent_dagger/sim_tool.py; S=outputs/sessions/s6_c60
    uv run $T init --session $S --scene baseline --seed 6 \
        --from-recording outputs/failure_sweep/baseline_s6_sa.npz --at-chunk 60
    uv run $T observe --session $S                       # renders + JSON of the scene
    uv run $T move --session $S --arm right --to 0.62 0.05 0.90 --pointing down --across bottle_3 --gripper 1
    uv run $T joints --session $S --targets <14 floats> --steps 30
    uv run $T hold --session $S --steps 15
    uv run $T undo --session $S

World frame: x forward (away from the robot bases, toward the back wall),
y to the robot's left, z up. Table top is z = 0.75. ``move`` targets are the
point between the fingertips. Actions are 14 joint targets: [left j1-j6,
left gripper, right j1-j6, right gripper], radians, gripper 0 = closed,
1 = open, one action per 1/30 s.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
import mujoco
import numpy as np
import tyro

from unseen_rollouts import CAMERAS, SceneName, make_scene

TASK = "put_plastic_bottles_in_bin"
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
ARM_SLICES = {"left": slice(0, 6), "right": slice(7, 13)}
GRIPPER_IDX = {"left": 6, "right": 13}
TCP_SITE = {"left": "left_tcp_site", "right": "right_tcp_site"}
POINTING = {"down": np.array([0.0, 0.0, -1.0]), "forward": np.array([1.0, 0.0, 0.0])}
FINGERTIP_OFFSET = 0.153  # fingertips lie this far past the tcp site along its pointing (+z) axis
FINGER_MAX_OPENING = 0.095  # metres between the jaws at gripper = 1
ARM_REACH = 0.62  # rough radius (m) from an arm's base within which top-down targets are reachable


# ---------------------------------------------------------------- session i/o

def _load_session(session: Path):
    meta = json.loads((session / "session.json").read_text())
    env, _ = make_scene(TASK, meta["scene"], meta["seed"])
    snap = np.load(session / f"state_{meta['step']:04d}.npz")["state"]
    mujoco.mj_setState(env.model, env.data, snap, STATE_SPEC)
    return env, meta


def _save_state(env, session: Path, step: int) -> None:
    buf = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
    mujoco.mj_getState(env.model, env.data, buf, STATE_SPEC)
    np.savez(session / f"state_{step:04d}.npz", state=buf)


def _append_actions(session: Path, states, actions, source: str) -> None:
    path = session / "actions.npz"
    if path.exists():
        old = np.load(path)
        states = np.concatenate([old["states"], states]); actions = np.concatenate([old["actions"], actions])
        sources = np.concatenate([old["source"], np.array([source] * (len(actions) - len(old["actions"])))])
    else:
        sources = np.array([source] * len(actions))
    np.savez(path, states=np.asarray(states, np.float32), actions=np.asarray(actions, np.float32), source=sources)


def _log(session: Path, entry: dict) -> None:
    with open(session / "log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------- scene report

def _site(env, arm: str):
    return mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE[arm])


def _subtree(m, root: int) -> list[int]:
    out, i = [root], 0
    while i < len(out):
        out += [b for b in range(m.nbody) if m.body_parentid[b] == out[i] and b != out[i]]
        i += 1
    return out


def _bottle_geometry(env, name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """(centre, long-axis unit vector, length) of a bottle from its collision geoms.

    The body origin sits at one end of the bottle mesh, so the centre is what a
    grasp must target, not the origin."""
    m, d = env.model, env.data
    b = m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")]
    R = d.xmat[b].reshape(3, 3)
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in _subtree(m, b) and (m.geom_contype[g] or m.geom_conaffinity[g])]
    pts = np.array([d.geom_xpos[g] for g in geoms])
    rb = np.array([m.geom_rbound[g] for g in geoms])[:, None]
    lo, hi = (pts - rb).min(0), (pts + rb).max(0)
    centre = (lo + hi) / 2
    local = (pts - centre) @ R
    spread = local.max(0) - local.min(0)
    k = int(np.argmax(spread))
    length = float(np.abs(hi - lo) @ np.abs(R[:, k]))  # bounding-box extent projected onto the long axis
    return centre, R[:, k], length


def _bottle_long_axis(env, name: str) -> tuple[np.ndarray, float]:
    _, axis, length = _bottle_geometry(env, name)
    return axis, length


def _r3(v) -> list[float]:
    return [round(float(x), 3) for x in v]


def _describe(env, meta) -> dict:
    m, d = env.model, env.data
    mujoco.mj_forward(m, d)
    state = env.get_state()
    result = env.evaluate_task().to_info(squeeze=True)
    bottles = []
    for i, name in enumerate(result["bottle_names"]):
        centre, axis, length = _bottle_geometry(env, name)
        bottles.append({"name": name, "centre_xyz": _r3(centre), "in_bin": bool(result["bottle_in_bin_mask"][i]),
                        "long_axis": _r3(axis), "length_m": round(length, 2),
                        "lying_flat": bool(abs(axis[2]) < 0.3)})
    bin_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "bin_joint")
    bin_pos = d.qpos[m.jnt_qposadr[bin_jid] : m.jnt_qposadr[bin_jid] + 3]
    arms = {}
    for arm in ("left", "right"):
        sid = _site(env, arm)
        R = d.site_xmat[sid].reshape(3, 3)
        tip = d.site_xpos[sid] + FINGERTIP_OFFSET * R[:, 2]
        base = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_link_1")]
        arms[arm] = {"fingertip_xyz": _r3(tip),
                     "pointing": [round(float(v), 2) for v in R[:, 2]],
                     "jaw_axis": [round(float(v), 2) for v in R[:, 1]],
                     "gripper_open": round(float(state[GRIPPER_IDX[arm]]), 2),
                     "jaw_gap_m": round(float(state[GRIPPER_IDX[arm]]) * FINGER_MAX_OPENING, 3),
                     "base_xy": _r3(base[:2]),
                     "joints_rad": _r3(state[ARM_SLICES[arm]])}
    return {
        "step": meta["step"], "robot_time_s": round(meta["step"] / 30, 2),
        "bottles_in_bin": f"{result['num_bottles_in_bin']}/{result['num_active_bottles']}",
        "bin": {"xyz": _r3(bin_pos), "opening_z": round(float(bin_pos[2] + result["bin_top_y"]), 3),
                "inner_radius_m": round(float(result["bin_radius_top"]), 3)},
        "bottles": bottles,
        "arms": arms,
        "notes": [f"table top z=0.75; each arm reaches roughly {ARM_REACH} m from its base_xy with the gripper pointing down",
                  f"jaws open to {FINGER_MAX_OPENING} m; bottles are ~0.06-0.07 m across, so grasp with the jaw_axis across the bottle's long_axis (use --across)",
                  "fingertip_xyz is the point between the fingertips; --to targets refer to it"],
        "state_14": _r3(state),
    }


def _render(env, session: Path, step: int) -> str:
    obs = env.get_obs()
    frame = np.concatenate([obs["images"][c].transpose(1, 2, 0) for c in CAMERAS], axis=1)
    path = session / f"obs_{step:04d}.png"
    imageio.imwrite(path, frame)
    return str(path)


# ---------------------------------------------------------------- acting

def _apply(env, session: Path, meta: dict, targets: np.ndarray, source: str) -> dict:
    """Step the sim through a (N, 14) sequence of joint targets, recording (s, a)."""
    states, actions = [], []
    for a in targets:
        states.append(env.get_state().copy()); actions.append(np.asarray(a, np.float32))
        env.step(np.asarray(a, np.float32), render_obs=False)
    meta["step"] += len(targets)
    _append_actions(session, np.array(states), np.array(actions), source)
    _save_state(env, session, meta["step"])
    (session / "session.json").write_text(json.dumps(meta))
    return _describe(env, meta)


def _interpolate(start: np.ndarray, goal: np.ndarray, steps: int) -> np.ndarray:
    t = np.linspace(0, 1, steps + 1)[1:, None]
    s = 3 * t**2 - 2 * t**3  # smooth-step: no velocity jump at either end
    return start[None, :] + (goal - start)[None, :] * s


def _solve_ik(env, arm: str, tip_target, target_z=None, target_y=None, iters: int = 400, damping: float = 0.05):
    """Damped-least-squares IK on this arm's 6 joints.

    ``tip_target`` is the fingertip point. ``target_z`` (pointing direction) and
    ``target_y`` (jaw-closing direction) constrain the orientation when given.
    Returns (joint targets, fingertip position error in metres)."""
    m = env.model
    d = mujoco.MjData(m)
    d.qpos[:] = env.data.qpos
    sid = _site(env, arm)
    qidx = np.array(env._qpos_indices)[ARM_SLICES[arm]]
    jids = [next(j for j in range(m.njnt) if m.jnt_qposadr[j] == q) for q in qidx]
    didx = np.array([m.jnt_dofadr[j] for j in jids])
    lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    tip_target = np.asarray(tip_target, float)
    for _ in range(iters):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        R = d.site_xmat[sid].reshape(3, 3)
        tip = d.site_xpos[sid] + FINGERTIP_OFFSET * R[:, 2]
        err = tip_target - tip
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        # Fingertip jacobian = site jacobian + rotation effect on the offset lever arm.
        lever = FINGERTIP_OFFSET * R[:, 2]
        jtip = jacp[:, didx] - np.cross(lever[None, :], jacr[:, didx].T).T
        rows, J = [err], [jtip]
        if target_z is not None:
            rows.append(0.5 * np.cross(R[:, 2], target_z)); J.append(jacr[:, didx])
        if target_y is not None:
            rows.append(0.5 * np.cross(R[:, 1], target_y)); J.append(jacr[:, didx])
        e = np.concatenate(rows); Jm = np.vstack(J)
        if np.linalg.norm(err) < 1e-3 and np.linalg.norm(e[3:]) < 1e-2:
            break
        dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + damping**2 * np.eye(len(e)), e)
        d.qpos[qidx] = np.clip(d.qpos[qidx] + np.clip(dq, -0.2, 0.2), lo, hi)
    mujoco.mj_kinematics(m, d)
    R = d.site_xmat[sid].reshape(3, 3)
    return d.qpos[qidx].copy(), float(np.linalg.norm(tip_target - (d.site_xpos[sid] + FINGERTIP_OFFSET * R[:, 2])))


def _plan_move(env, cmd) -> tuple[np.ndarray, dict]:
    """Resolve a Move into joint targets, falling back to looser orientation constraints if needed."""
    sid = _site(env, cmd.arm)
    R = env.data.site_xmat[sid].reshape(3, 3)
    target_z = POINTING.get(cmd.pointing)  # None for "keep"
    target_y = None
    if cmd.across:
        axis, _ = _bottle_long_axis(env, cmd.across)
        z = target_z if target_z is not None else R[:, 2]
        y = np.cross(z, axis)
        if np.linalg.norm(y) < 1e-3:
            y = R[:, 1]
        target_y = y / np.linalg.norm(y)
        if np.dot(target_y, R[:, 1]) < 0:  # pick the jaw direction closest to the current one (symmetric grasp)
            target_y = -target_y
    attempts = [("pointing+jaws", target_z, target_y), ("pointing only", target_z, None), ("position only", None, None)]
    for label, tz, ty in attempts:
        q, err = _solve_ik(env, cmd.arm, cmd.to, tz, ty)
        if err < 0.02:
            return q, {"ik_position_error_m": round(err, 4), "ik_constraints": label}
    return q, {"ik_position_error_m": round(err, 4), "ik_constraints": "position only",
               "warning": f"fingertip target unreachable for the {cmd.arm} arm (closest approach {err:.2f} m); try the other arm or a nearer point"}


# ---------------------------------------------------------------- commands

@dataclass
class Init:
    """Start a session: build the scene and optionally fast-forward through a recording."""
    session: str
    scene: SceneName = "baseline"
    seed: int = 0
    from_recording: str | None = None  # <name>_sa.npz from unseen_rollouts.py
    at_chunk: int = 0  # replay this many 15-action chunks of the recording first


@dataclass
class Observe:
    """Render the cameras and print a JSON description of the scene."""
    session: str


@dataclass
class Move:
    """Move one gripper so its fingertips reach a world xyz target over --steps actions (IK -> joint targets)."""
    session: str
    arm: Literal["left", "right"]
    to: tuple[float, float, float]  # fingertip target, world frame (table top is z=0.75)
    pointing: Literal["down", "forward", "keep"] = "keep"
    across: str | None = None  # e.g. "bottle_3": turn the jaws to close across that bottle's long axis
    gripper: float | None = None  # 0 closed .. 1 open; None keeps the current opening
    steps: int = 30


@dataclass
class Joints:
    """Send raw joint targets (14 floats) reached smoothly over --steps actions."""
    session: str
    targets: tuple[float, ...]
    steps: int = 30


@dataclass
class Hold:
    """Hold the current pose for --steps actions (lets objects settle or a grasp close)."""
    session: str
    steps: int = 15


@dataclass
class Undo:
    """Return to the state before the last command (the recorded actions are kept, marked undone)."""
    session: str


def main(cmd) -> None:
    session = Path(cmd.session)
    t0 = time.time()
    if isinstance(cmd, Init):
        session.mkdir(parents=True, exist_ok=True)
        for old in session.glob("*"):
            old.unlink()
        env, _ = make_scene(TASK, cmd.scene, cmd.seed)
        meta = {"scene": cmd.scene, "seed": cmd.seed, "step": 0, "history": []}
        if cmd.from_recording:
            rec = np.load(cmd.from_recording)
            n = int(cmd.at_chunk * 15)
            for a in rec["actions"][:n]:
                env.step(a, render_obs=False)
            meta["step"] = n
            meta["replayed_from"] = cmd.from_recording
        _save_state(env, session, meta["step"])
        (session / "session.json").write_text(json.dumps(meta))
        report = _describe(env, meta)
        report["image"] = _render(env, session, meta["step"])
    elif isinstance(cmd, Undo):
        meta = json.loads((session / "session.json").read_text())
        if not meta["history"]:
            print(json.dumps({"error": "nothing to undo"})); return
        meta["step"] = meta["history"].pop()
        (session / "session.json").write_text(json.dumps(meta))
        env, meta = _load_session(session)
        report = _describe(env, meta); report["image"] = _render(env, session, meta["step"])
        _log(session, {"cmd": "undo", "step": meta["step"]})
    else:
        env, meta = _load_session(session)
        if isinstance(cmd, Observe):
            report = _describe(env, meta); report["image"] = _render(env, session, meta["step"])
        else:
            meta["history"].append(meta["step"])
            start = env.get_state().copy()
            if isinstance(cmd, Hold):
                targets = np.tile(start, (cmd.steps, 1)); source = "agent:hold"; extra = {}
            elif isinstance(cmd, Joints):
                goal = np.asarray(cmd.targets, np.float64)
                if goal.shape != (14,):
                    print(json.dumps({"error": f"need 14 targets, got {goal.shape}"})); return
                targets = _interpolate(start, goal, cmd.steps); source = "agent:joints"; extra = {}
            else:  # Move
                q, extra = _plan_move(env, cmd)
                goal = start.copy(); goal[ARM_SLICES[cmd.arm]] = q
                if cmd.gripper is not None:
                    goal[GRIPPER_IDX[cmd.arm]] = float(np.clip(cmd.gripper, 0, 1))
                targets = _interpolate(start, goal, cmd.steps); source = f"agent:move_{cmd.arm}"
            report = _apply(env, session, meta, targets, source)
            report.update(extra)
            report["image"] = _render(env, session, meta["step"])
            _log(session, {"cmd": type(cmd).__name__.lower(), "args": {k: v for k, v in vars(cmd).items() if k != "session"},
                           "step": meta["step"], **extra})
    report["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(tyro.cli(Init | Observe | Move | Joints | Hold | Undo))
