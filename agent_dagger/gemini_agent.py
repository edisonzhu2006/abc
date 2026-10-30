"""Gemini-backed monitor and corrector, running in-process against a sim session.

The corrector is a function-calling loop: the model sees the scene report and
the three camera images, calls one tool at a time (observe / move / grasp /
place / hold / undo / handback), gets the new report and image back, and
repeats until it hands back. Every call, with the model's own reasoning
summary, is logged to <session>/gemini_trace.jsonl, and the actions are
recorded in the session like any other corrector.

The API key is read from GEMINI_API_KEY or ~/.config/agent_dagger/env (never
from the repo).

    uv run agent_dagger/gemini_agent.py correct --session outputs/sessions/s6_c28 --reason "no progress 5 s"
    uv run agent_dagger/gemini_agent.py monitor --image outputs/.../obs_0420.png --chunk 28
"""

from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import tyro

from sim_tool import (STATE_SPEC, _describe, _load_session, _log, _render, run_grasp, run_hold, run_move, run_place)
from unseen_rollouts import CAMERAS

# gemini-4-argon and the 3.x pro models need a billed project (free-tier quota is 0);
# gemini-3.5-flash works on the free tier and answered the monitor test correctly.
DEFAULT_MODEL = "gemini-3.5-flash"


def load_key() -> None:
    if os.environ.get("GEMINI_API_KEY"):
        return
    env = Path.home() / ".config" / "agent_dagger" / "env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k and v and k not in os.environ:
                os.environ[k] = v
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY not set; put it in ~/.config/agent_dagger/env")


def _png_part(path_or_array):
    from google.genai import types

    if isinstance(path_or_array, (str, Path)):
        data = Path(path_or_array).read_bytes()
    else:
        buf = io.BytesIO(); imageio.imwrite(buf, path_or_array, format="png"); data = buf.getvalue()
    return types.Part.from_bytes(data=data, mime_type="image/png")


# ---------------------------------------------------------------- monitor

MONITOR_PROMPT = """You are the human-in-the-loop monitor for a DAgger robot data-collection run.
A learned policy is driving two i2rt YAM arms (bimanual, parallel-jaw grippers) on a tabletop; the task is
"{prompt}". The image shows, left to right: the overhead camera, the left wrist camera, the right wrist camera.
Time elapsed: {seconds:.0f} s. {history}
Decide whether an expert should take over RIGHT NOW. Intervene when the policy is failing or about to fail in a way it
is unlikely to recover from: repeated missed grasps (jaws closing on nothing, bottle pushed around), a lifted bottle
dropped outside the bin, an arm hovering or oscillating without progress for ~5 s, or the task made impossible.
Do not intervene for steady progress or a single clean retry."""

MONITOR_SCHEMA = {"type": "object", "properties": {"intervene": {"type": "boolean"}, "failure_mode": {"type": "string"},
                                                     "reason": {"type": "string"}}, "required": ["intervene", "failure_mode", "reason"]}


def gemini_monitor(image, prompt: str, seconds: float, history: str = "", model: str = DEFAULT_MODEL) -> dict:
    from google import genai
    from google.genai import types

    load_key()
    client = genai.Client()
    resp = client.models.generate_content(
        model=model,
        contents=[_png_part(image), MONITOR_PROMPT.format(prompt=prompt, seconds=seconds, history=history)],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=MONITOR_SCHEMA),
    )
    return json.loads(resp.text)


# ---------------------------------------------------------------- corrector

CORRECTOR_SYSTEM = """You are the expert corrector in a DAgger loop for a bimanual robot (two i2rt YAM arms, 6 joints +
parallel-jaw gripper each, bases at the near table edge: left base (0.25, +0.31), right base (0.25, -0.31)).
A learned policy stalled; the monitor flagged it; you take over, fix the flagged failure, and hand control back as soon as
the policy can plausibly continue. Task: put every plastic bottle into the bin.

World frame: x forward (away from the bases), y to the robot's left, z up; table top z = 0.75. Each arm reaches about
0.62 m from its base. Work out from the report which arm can reach the bin and which bottles; a bottle reachable only by
the arm that cannot reach the bin can be handed off: set it down around (0.52, -0.08) and pick it up with the other arm.

Tools act on the live simulation and return the scene report plus a fresh camera image. Prefer the macros:
grasp(arm, bottle) does pre-grasp, hover check, descend, close, verify and lift; place(arm) carries high over the bin,
releases and backs the arm out. Use move for anything else (parking an arm, nudging a bottle, routing around the bin:
never sweep an arm through the bin's column while a bottle stands in it). After each tool result, check what actually
happened (jaw_gap_m ~0.05 means a bottle is held; 0.0 means the jaws closed on nothing; bottles_in_bin is the score).

Give every tool call a one-sentence note: which bottle, what you saw, what you expect. When the flagged failure is fixed
(typically one stuck bottle binned) and both arms are parked clear of the bin, high, open and toward their own sides,
call handback with a note saying why the policy can continue. Also hand back if 25 calls pass without progress.
Do not try to finish every bottle yourself."""


def _tools():
    from google.genai import types

    num = {"type": "NUMBER"}
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(name="observe", description="Report the scene and return a fresh camera image; no motion.",
                                  parameters={"type": "OBJECT", "properties": {}}),
        types.FunctionDeclaration(name="grasp", description="Macro: pick up a lying bottle with one arm (pre-grasp, hover check, descend, close, verify, lift).",
                                  parameters={"type": "OBJECT", "properties": {"arm": {"type": "STRING", "enum": ["left", "right"]},
                                                                               "bottle": {"type": "STRING"}, "note": {"type": "STRING"}},
                                              "required": ["arm", "bottle", "note"]}),
        types.FunctionDeclaration(name="place", description="Macro: carry the held bottle high over the bin centre, release it, back the arm out open.",
                                  parameters={"type": "OBJECT", "properties": {"arm": {"type": "STRING", "enum": ["left", "right"]},
                                                                               "note": {"type": "STRING"}}, "required": ["arm", "note"]}),
        types.FunctionDeclaration(name="move", description="Move one arm so its grasp point (finger-pad centre) reaches x,y,z over `steps` actions (30 = 1 s).",
                                  parameters={"type": "OBJECT", "properties": {
                                      "arm": {"type": "STRING", "enum": ["left", "right"]}, "x": num, "y": num, "z": num,
                                      "pointing": {"type": "STRING", "description": "'down', 'forward', 'keep', or an explicit vector 'x,y,z'"},
                                      "across": {"type": "STRING", "description": "bottle name to turn the jaws across, e.g. bottle_3"},
                                      "gripper": {"type": "NUMBER", "description": "1 = open, 0 = closed; omit to keep the current command"},
                                      "tilt_tolerance_deg": num, "steps": {"type": "INTEGER"}, "note": {"type": "STRING"}},
                                      "required": ["arm", "x", "y", "z", "note"]}),
        types.FunctionDeclaration(name="hold", description="Hold the current targets for `steps` actions.",
                                  parameters={"type": "OBJECT", "properties": {"steps": {"type": "INTEGER"}, "note": {"type": "STRING"}}, "required": ["note"]}),
        types.FunctionDeclaration(name="undo", description="Return to the state before the last command.",
                                  parameters={"type": "OBJECT", "properties": {}}),
        types.FunctionDeclaration(name="handback", description="End the intervention and return control to the policy.",
                                  parameters={"type": "OBJECT", "properties": {"note": {"type": "STRING"}}, "required": ["note"]}),
    ])


def _compact(report: dict) -> dict:
    """Trim the report for the model: drop verbose fields it does not need every turn."""
    r = {k: v for k, v in report.items() if k not in ("state_14", "image", "seconds")}
    for arm in r.get("arms", {}).values():
        arm.pop("joints_rad", None)
    return r


def run_corrector(session: Path, reason: str, model: str = DEFAULT_MODEL, max_calls: int = 40, thinking: str = "high") -> dict:
    from google import genai
    from google.genai import types

    load_key()
    client = genai.Client()
    env, meta = _load_session(session)
    m, d = env.model, env.data
    trace = open(session / "gemini_trace.jsonl", "a")

    def execute(name: str, args: dict) -> tuple[dict, bool]:
        nonlocal meta
        if name == "observe":
            return _describe(env, meta), False
        if name == "grasp":
            return run_grasp(env, session, meta, args["arm"], args["bottle"], args.get("note", "")), False
        if name == "place":
            return run_place(env, session, meta, args["arm"], args.get("note", "")), False
        if name == "move":
            kw = {"arm": args["arm"], "to": (float(args["x"]), float(args["y"]), float(args["z"])), "note": args.get("note", "")}
            for k in ("pointing", "across", "tilt_tolerance_deg", "steps"):
                if args.get(k) is not None:
                    kw[k] = args[k]
            if args.get("gripper") is not None:
                kw["gripper"] = float(args["gripper"])
            return run_move(env, session, meta, **kw), False
        if name == "hold":
            return run_hold(env, session, meta, int(args.get("steps", 15)), args.get("note", "")), False
        if name == "undo":
            if not meta["history"]:
                return {"error": "nothing to undo"}, False
            meta["step"] = meta["history"].pop()
            snap = np.load(session / f"state_{meta['step']:04d}.npz")["state"]
            mujoco.mj_setState(m, d, snap, STATE_SPEC); mujoco.mj_forward(m, d)
            (session / "session.json").write_text(json.dumps(meta))
            _log(session, {"cmd": "undo", "step": meta["step"]})
            return _describe(env, meta), False
        if name == "handback":
            meta["handed_back_at"] = meta["step"]
            (session / "session.json").write_text(json.dumps(meta))
            _log(session, {"cmd": "handback", "step": meta["step"], "note": args.get("note", "")})
            return {"handed_back": True, "step": meta["step"]}, True
        return {"error": f"unknown tool {name}"}, False

    config = types.GenerateContentConfig(
        system_instruction=CORRECTOR_SYSTEM, tools=[_tools()],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_level=thinking),
    )
    chat = client.chats.create(model=model, config=config)
    first = _describe(env, meta)
    img = _render(env, session, meta["step"])
    message = [types.Part.from_text(text=f"The monitor flagged: {reason}\nCurrent scene (JSON):\n{json.dumps(_compact(first))}\n"
                                         "Camera image attached (overhead | left wrist | right wrist). Fix the failure, then hand back."),
               _png_part(img)]
    calls, done, t0 = 0, False, time.time()
    summary = {"start_in_bin": first["bottles_in_bin"], "calls": 0}
    while not done and calls < max_calls:
        resp = chat.send_message(message)
        thoughts = [p.text for p in (resp.candidates[0].content.parts or []) if getattr(p, "thought", False) and p.text]
        fcs = [p.function_call for p in (resp.candidates[0].content.parts or []) if p.function_call]
        texts = [p.text for p in (resp.candidates[0].content.parts or []) if p.text and not getattr(p, "thought", False)]
        if not fcs:
            # No tool call: nudge once, then stop.
            trace.write(json.dumps({"calls": calls, "text": texts, "thoughts": thoughts}) + "\n"); trace.flush()
            message = "Call a tool (observe, grasp, place, move, hold, undo) or handback with a note."
            calls += 1
            continue
        responses = []
        for fc in fcs:
            args = dict(fc.args or {})
            calls += 1
            t = time.time()
            result, done = execute(fc.name, args)
            entry = {"call": calls, "tool": fc.name, "args": args, "step": meta["step"], "seconds": round(time.time() - t, 1),
                     "result": {k: result[k] for k in ("bottles_in_bin", "grasp", "place", "error", "warning", "ik_constraints") if k in result},
                     "thoughts": thoughts, "text": texts}
            trace.write(json.dumps(entry) + "\n"); trace.flush()
            print(f"[{calls:2d}] {fc.name}({', '.join(f'{k}={v}' for k, v in args.items() if k != 'note')})  note: {args.get('note', '')[:90]}"
                  + (f"  -> {entry['result']}" if entry["result"] else ""), flush=True)
            responses.append(types.Part.from_function_response(name=fc.name, response={"result": _compact(result) if isinstance(result, dict) else result}))
            if done:
                break
        message = responses
        if not done and "error" not in result:
            message = responses + [_png_part(_render(env, session, meta["step"]))]
    final = _describe(env, meta)
    summary.update(calls=calls, end_in_bin=final["bottles_in_bin"], handed_back=done, wall_s=round(time.time() - t0), model=model)
    trace.write(json.dumps({"summary": summary}) + "\n"); trace.close(); env.close()
    print("corrector done:", json.dumps(summary), flush=True)
    return summary


# ---------------------------------------------------------------- CLI

@dataclass
class Correct:
    """Run the Gemini corrector on an existing session until it hands back."""
    session: str
    reason: str = "no progress"
    model: str = DEFAULT_MODEL
    max_calls: int = 40
    thinking: str = "high"


@dataclass
class Monitor:
    """Ask Gemini whether to intervene, given one camera image."""
    image: str
    chunk: int = 0
    prompt: str = "throw plastic bottles in bin"
    model: str = DEFAULT_MODEL


def main(cmd) -> None:
    if isinstance(cmd, Correct):
        run_corrector(Path(cmd.session), cmd.reason, cmd.model, cmd.max_calls, cmd.thinking)
    else:
        print(json.dumps(gemini_monitor(cmd.image, cmd.prompt, cmd.chunk * 0.5, model=cmd.model), indent=1))


if __name__ == "__main__":
    main(tyro.cli(Correct | Monitor))
