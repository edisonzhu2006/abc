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
grasp point (centre of the finger pads, 6 cm from the wrist; the fingertips
reach 9 cm further). Actions are 14 joint targets: [left j1-j6,
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
GRASP_OFFSET = 0.06  # the finger pads' centre lies this far past the tcp site: the point that should meet an object
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
        grasp = d.site_xpos[sid] + GRASP_OFFSET * R[:, 2]
        tip = d.site_xpos[sid] + FINGERTIP_OFFSET * R[:, 2]
        base = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_link_1")]
        arms[arm] = {"grasp_point_xyz": _r3(grasp),
                     "fingertip_xyz": _r3(tip),
                     "pointing": [round(float(v), 2) for v in R[:, 2]],
                     "jaw_axis": [round(float(v), 2) for v in R[:, 1]],
                     "gripper_open": round(float(state[GRIPPER_IDX[arm]]), 2),
                     "gripper_cmd": round(float(meta.get("gripper_cmd", {}).get(arm, state[GRIPPER_IDX[arm]])), 2),
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
        "notes": [f"table top z=0.75; each arm reaches roughly {ARM_REACH} m from its base_xy",
                  "--to targets refer to grasp_point_xyz (the finger pads' centre); the fingertips extend ~0.09 m further along 'pointing'",
                  "to grasp a bottle lying on the table: move the grasp point to the bottle's centre_xyz raised by ~0.07 m, gripper open, "
                  "pointing down with --across <bottle>, then close (gripper 0) and lift; a closed jaw_gap_m of ~0.05-0.06 means a bottle is held",
                  f"jaws open to {FINGER_MAX_OPENING} m; bottles are ~0.06-0.07 m across"],
        "state_14": _r3(state),
    }


def _render(env, session: Path, step: int) -> str:
    obs = env.get_obs()
    frame = np.concatenate([obs["images"][c].transpose(1, 2, 0) for c in CAMERAS], axis=1)
    path = session / f"obs_{step:04d}.png"
    imageio.imwrite(path, frame)
    return str(path)


# ---------------------------------------------------------------- acting

def _commanded_start(env, meta: dict) -> np.ndarray:
    """Current joint state, but with the grippers at their last *commanded* values.

    After closing on an object the measured opening is wider than the command
    (the fingers stop at the object); re-commanding the measured value would
    drop the squeeze force and release it."""
    start = env.get_state().copy()
    for arm, idx in GRIPPER_IDX.items():
        start[idx] = meta.get("gripper_cmd", {}).get(arm, start[idx])
    return start


def _apply(env, session: Path, meta: dict, targets: np.ndarray, source: str) -> dict:
    """Step the sim through a (N, 14) sequence of joint targets, recording (s, a)."""
    states, actions = [], []
    meta["gripper_cmd"] = {arm: float(targets[-1][idx]) for arm, idx in GRIPPER_IDX.items()}
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


def _cone_error(cur: np.ndarray, target: np.ndarray, tol_deg: float) -> np.ndarray:
    """Rotation (small-angle vector) that brings ``cur`` to within ``tol_deg`` of ``target``; zero if already inside."""
    axis = np.cross(cur, target)
    s = np.linalg.norm(axis)
    angle = np.arctan2(s, np.dot(cur, target))
    if angle <= np.radians(tol_deg) or s < 1e-9:
        return np.zeros(3)
    return axis / s * (angle - np.radians(tol_deg))


def _solve_ik(env, arm: str, tip_target, target_z=None, across_axis=None, tilt_tol_deg: float = 35.0,
              hold_jaw=None, iters: int = 400, damping: float = 0.05, restarts: int = 4, seed: int = 0):
    """Damped-least-squares IK on this arm's 6 joints, best of a few restarts.

    ``tip_target`` is the grasp point (finger-pad centre). ``target_z`` keeps the pointing
    direction within ``tilt_tol_deg`` of that vector (a cone, not an exact
    direction: this arm cannot point straight down over much of the table).
    ``across_axis`` turns the jaw-closing axis perpendicular to that vector (a
    bottle's long axis) with the smallest rotation; ``hold_jaw`` instead keeps
    the jaw axis along that world vector (no wrist twist while carrying).
    Returns (joint targets, grasp-point position error in metres)."""
    m = env.model
    d = mujoco.MjData(m)
    sid = _site(env, arm)
    qidx = np.array(env._qpos_indices)[ARM_SLICES[arm]]
    jids = [next(j for j in range(m.njnt) if m.jnt_qposadr[j] == q) for q in qidx]
    didx = np.array([m.jnt_dofadr[j] for j in jids])
    lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    tip_target = np.asarray(tip_target, float)
    rng = np.random.default_rng(seed)
    best_q, best_err = None, np.inf
    for restart in range(restarts):
        d.qpos[:] = env.data.qpos
        if restart > 0:
            d.qpos[qidx] = rng.uniform(lo, hi)
        for _ in range(iters):
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            R = d.site_xmat[sid].reshape(3, 3)
            tip = d.site_xpos[sid] + GRASP_OFFSET * R[:, 2]
            err = tip_target - tip
            mujoco.mj_jacSite(m, d, jacp, jacr, sid)
            # Grasp-point jacobian = site jacobian + rotation effect on the offset lever arm.
            lever = GRASP_OFFSET * R[:, 2]
            jtip = jacp[:, didx] - np.cross(lever[None, :], jacr[:, didx].T).T
            rows, J = [err], [jtip]
            if target_z is not None:
                rows.append(0.5 * _cone_error(R[:, 2], np.asarray(target_z, float), tilt_tol_deg)); J.append(jacr[:, didx])
            if across_axis is not None:
                y = R[:, 1]
                perp = y - np.dot(y, across_axis) * across_axis
                perp /= max(np.linalg.norm(perp), 1e-9)
                rows.append(0.5 * np.cross(y, perp)); J.append(jacr[:, didx])
            elif hold_jaw is not None:
                rows.append(0.5 * np.cross(R[:, 1], hold_jaw)); J.append(jacr[:, didx])
            e = np.concatenate(rows); Jm = np.vstack(J)
            if np.linalg.norm(err) < 1e-3 and np.linalg.norm(e[3:]) < 1e-2:
                break
            dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + damping**2 * np.eye(len(e)), e)
            d.qpos[qidx] = np.clip(d.qpos[qidx] + np.clip(dq, -0.2, 0.2), lo, hi)
        mujoco.mj_kinematics(m, d)
        R = d.site_xmat[sid].reshape(3, 3)
        pos_err = float(np.linalg.norm(tip_target - (d.site_xpos[sid] + GRASP_OFFSET * R[:, 2])))
        ori_err = float(np.linalg.norm(np.concatenate(rows[1:]))) if len(rows) > 1 else 0.0
        # Prefer the solution nearest the current pose: a random restart must be
        # clearly better to win, so nearby targets get consistent, predictable poses.
        joint_dist = float(np.linalg.norm(d.qpos[qidx] - env.data.qpos[qidx]))
        score = pos_err + 0.05 * ori_err + 0.01 * joint_dist
        if score < best_err:
            best_q, best_err, best_pos = d.qpos[qidx].copy(), score, pos_err
        if best_pos < 2e-3 and ori_err < 1e-2:
            break
    return best_q, best_pos


def _plan_move(env, cmd) -> tuple[np.ndarray, dict]:
    """Resolve a Move into joint targets, falling back to looser orientation constraints if needed."""
    if "," in cmd.pointing:
        target_z = np.array([float(v) for v in cmd.pointing.split(",")]); target_z /= np.linalg.norm(target_z)
    else:
        target_z = POINTING.get(cmd.pointing)  # None for "keep"
    across = _bottle_long_axis(env, cmd.across)[0] if cmd.across else None
    # Without --across, keep the jaws where they are so a held object is not twisted out.
    jaw_now = env.data.site_xmat[_site(env, cmd.arm)].reshape(3, 3)[:, 1].copy()
    tol = f"pointing within {cmd.tilt_tolerance_deg:.0f} deg"
    if across is not None and abs(cmd.jaw_angle_deg - 90.0) > 0.5:
        # Oblique grasp: aim the jaw axis at the requested angle to the bottle, in the
        # plane perpendicular to the pointing direction (nearest of the two mirror options).
        z = target_z if target_z is not None else env.data.site_xmat[_site(env, cmd.arm)].reshape(3, 3)[:, 2]
        perp = np.cross(z, across); perp /= max(np.linalg.norm(perp), 1e-9)
        along = across - np.dot(across, z) * z; along /= max(np.linalg.norm(along), 1e-9)
        th = np.radians(cmd.jaw_angle_deg)
        candidates = [s1 * np.sin(th) * perp + s2 * np.cos(th) * along for s1 in (1, -1) for s2 in (1, -1)]
        jaw_target = max(candidates, key=lambda v: np.dot(v, jaw_now))
        attempts = [(f"{tol} + jaws at {cmd.jaw_angle_deg:.0f} deg to {cmd.across}", target_z, None, jaw_target)]
    elif across is not None:
        attempts = [(f"{tol} + jaws across {cmd.across}", target_z, across, None)]
    else:
        attempts = [(f"{tol} + jaws held", target_z, None, jaw_now)]
    attempts += [(tol, target_z, None, None), ("position only", None, None, None)]
    for label, tz, ax, hj in attempts:
        q, err = _solve_ik(env, cmd.arm, cmd.to, tz, ax, tilt_tol_deg=cmd.tilt_tolerance_deg, hold_jaw=hj)
        if err < 0.02:
            info = {"ik_position_error_m": round(err, 4), "ik_constraints": label}
            if across is not None:
                # Jaw axis vs bottle axis at the solved pose: 90 deg = jaws exactly across the bottle.
                d2 = mujoco.MjData(env.model); d2.qpos[:] = env.data.qpos
                d2.qpos[np.array(env._qpos_indices)[ARM_SLICES[cmd.arm]]] = q
                mujoco.mj_kinematics(env.model, d2)
                jaw = d2.site_xmat[_site(env, cmd.arm)].reshape(3, 3)[:, 1]
                info["jaw_to_bottle_deg"] = round(float(np.degrees(np.arccos(abs(np.dot(jaw, across))))), 1)
            return q, info
    return q, {"ik_position_error_m": round(err, 4), "ik_constraints": "position only",
               "warning": f"grasp-point target unreachable for the {cmd.arm} arm (closest approach {err:.2f} m); try the other arm or a nearer point"}


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
    """Move one gripper so its grasp point (finger-pad centre) reaches a world xyz target over --steps actions (IK -> joint targets)."""
    session: str
    arm: Literal["left", "right"]
    to: tuple[float, float, float]  # grasp-point target, world frame (table top is z=0.75)
    pointing: str = "keep"  # "down", "forward", "keep", or an explicit world vector "x,y,z" (e.g. "0.45,0,-0.89")
    across: str | None = None  # e.g. "bottle_3": turn the jaws to close across that bottle's long axis
    jaw_angle_deg: float = 90.0  # with --across: angle between the jaw-closing axis and the bottle's long axis (90 = straight across)
    tilt_tolerance_deg: float = 35.0  # how far from the --pointing direction the gripper may tilt
    gripper: float | None = None  # 0 closed .. 1 open; None keeps the current opening
    steps: int = 30
    note: str = ""  # why you are doing this (recorded with the action, shown in replays)


@dataclass
class Joints:
    """Send raw joint targets (14 floats) reached smoothly over --steps actions."""
    session: str
    targets: tuple[float, ...]
    steps: int = 30
    note: str = ""  # why you are doing this (recorded with the action, shown in replays)


@dataclass
class Hold:
    """Hold the current pose for --steps actions (lets objects settle or a grasp close)."""
    session: str
    steps: int = 15
    note: str = ""  # why you are doing this (recorded with the action, shown in replays)


@dataclass
class Undo:
    """Return to the state before the last command (the recorded actions are kept, marked undone)."""
    session: str


@dataclass
class Handback:
    """End the intervention and return control to the policy, stating why."""
    session: str
    note: str = ""


def main(cmd) -> None:
    session = Path(cmd.session)
    t0 = time.time()
    if isinstance(cmd, Init):
        session.mkdir(parents=True, exist_ok=True)
        for old in session.glob("*"):
            old.unlink()
        env, _ = make_scene(TASK, cmd.scene, cmd.seed)
        meta = {"scene": cmd.scene, "seed": cmd.seed, "step": 0, "history": []}
        last = env.get_state()
        if cmd.from_recording:
            rec = np.load(cmd.from_recording)
            n = int(cmd.at_chunk * 15)
            for a in rec["actions"][:n]:
                env.step(a, render_obs=False)
            meta["step"] = n
            meta["replayed_from"] = cmd.from_recording
            if n:
                last = rec["actions"][n - 1]
        meta["gripper_cmd"] = {arm: float(last[idx]) for arm, idx in GRIPPER_IDX.items()}
        _save_state(env, session, meta["step"])
        (session / "session.json").write_text(json.dumps(meta))
        report = _describe(env, meta)
        report["image"] = _render(env, session, meta["step"])
    elif isinstance(cmd, Handback):
        meta = json.loads((session / "session.json").read_text())
        meta["handed_back_at"] = meta["step"]
        (session / "session.json").write_text(json.dumps(meta))
        _log(session, {"cmd": "handback", "step": meta["step"], "note": cmd.note})
        env, meta = _load_session(session)
        report = _describe(env, meta); report["handed_back"] = True
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
            start = _commanded_start(env, meta)
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
    main(tyro.cli(Init | Observe | Move | Joints | Hold | Undo | Handback))
