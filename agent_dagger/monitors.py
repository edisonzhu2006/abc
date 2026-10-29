"""Intervention monitors: given what the robot sees, decide whether to take over.

A monitor is any callable ``(MonitorContext) -> Decision``. ``StallMonitor``
uses the simulator's own task evaluator (privileged ground truth) and needs no
API; it is the stand-in until Astra is wired up, and later the yardstick Astra's
calls are scored against. ``AstraMonitor`` asks GPT-6 Astra from camera frames.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class MonitorContext:
    chunk: int  # chunks executed so far (each = 15 actions = 0.5 s)
    images: dict[str, np.ndarray]  # camera name -> (3, H, W) uint8
    state: np.ndarray  # (14,) joint positions, policy layout
    prompt: str
    task_eval: dict[str, Any] = field(default_factory=dict)  # sim ground truth; Astra never sees it


@dataclass
class Decision:
    intervene: bool
    reason: str
    raw: Any = None  # full monitor output, kept for the log


class Monitor(Protocol):
    name: str

    def __call__(self, ctx: MonitorContext) -> Decision: ...


class StallMonitor:
    """Intervene when no bottle has gone into the bin for ``patience`` chunks."""

    name = "stall"

    def __init__(self, patience: int = 20):
        self.patience = patience
        self._best = 0
        self._last_progress_chunk = 0

    def __call__(self, ctx: MonitorContext) -> Decision:
        in_bin = int(ctx.task_eval.get("num_bottles_in_bin", 0))
        total = int(ctx.task_eval.get("num_active_bottles", 0))
        if in_bin > self._best:
            self._best, self._last_progress_chunk = in_bin, ctx.chunk
        stalled = ctx.chunk - self._last_progress_chunk
        if total and in_bin == total:
            return Decision(False, f"done: {in_bin}/{total} in bin")
        if stalled >= self.patience:
            # Restart the clock so the next call gives the policy a fresh window
            # instead of re-firing on every check.
            self._last_progress_chunk = ctx.chunk
            return Decision(True, f"no progress for {stalled} chunks ({stalled / 2:.0f} s), "
                                  f"stuck at {in_bin}/{total} in bin")
        return Decision(False, f"{in_bin}/{total} in bin, last progress {stalled} chunks ago")


_ASTRA_INSTRUCTIONS = """\
You are monitoring a learned robot policy for a DAgger data-collection loop and
must decide whether a human-level operator should take over right now.

Robot: two i2rt YAM arms (bimanual), each 6 joints + parallel gripper, on a
tabletop. Images: "top" is a fixed camera looking down at the table; "left" and
"right" are wrist cameras on each arm. The policy is executing the task:
"{prompt}". Time elapsed: {seconds:.0f} s ({chunk} chunks of 0.5 s).

Intervene only if the policy is failing or about to fail in a way it is
unlikely to recover from on its own (stuck, repeating a failed motion, object
knocked out of reach, wrong object, task made impossible). Do not intervene for
slow but steady progress.
"""

_ASTRA_SCHEMA = {
    "type": "object",
    "properties": {
        "intervene": {"type": "boolean"},
        "reason": {"type": "string"},
        "failure_mode": {"type": "string"},
    },
    "required": ["intervene", "reason", "failure_mode"],
    "additionalProperties": False,
}


def _png_base64(chw: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(chw.transpose(1, 2, 0))).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _png_data_url(chw: np.ndarray) -> str:
    return "data:image/png;base64," + _png_base64(chw)


class AstraMonitor:
    """Ask GPT-6 Astra (OpenAI Responses API) whether to intervene.

    Needs ``pip install openai`` and OPENAI_API_KEY. Untested against the live
    API as of 2026-10-29 (no key available when this was written)."""

    name = "astra"

    def __init__(self, model: str = "gpt-6-astra", effort: str = "low"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.effort = effort

    def __call__(self, ctx: MonitorContext) -> Decision:
        text = _ASTRA_INSTRUCTIONS.format(prompt=ctx.prompt, seconds=ctx.chunk * 0.5, chunk=ctx.chunk)
        content = [{"type": "input_text", "text": text}]
        for cam, img in ctx.images.items():
            content.append({"type": "input_text", "text": f"camera: {cam}"})
            content.append({"type": "input_image", "image_url": _png_data_url(img)})
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": self.effort},
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": "intervention_decision",
                             "schema": _ASTRA_SCHEMA, "strict": True}},
        )
        out = json.loads(response.output_text)
        return Decision(bool(out["intervene"]), f"{out['failure_mode']}: {out['reason']}", raw=out)


class ClaudeMonitor:
    """Ask Claude (Anthropic Messages API) whether to intervene, from the camera frames.

    Needs the ``anthropic`` package and ANTHROPIC_API_KEY (a Console API key;
    a claude.ai subscription does not cover API calls). Same prompt and output
    schema as AstraMonitor so the two are interchangeable in the loop."""

    name = "claude"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def __call__(self, ctx: MonitorContext) -> Decision:
        text = _ASTRA_INSTRUCTIONS.format(prompt=ctx.prompt, seconds=ctx.chunk * 0.5, chunk=ctx.chunk)
        content = []
        for cam, img in ctx.images.items():
            content.append({"type": "text", "text": f"camera: {cam}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": _png_base64(img)}})
        content.append({"type": "text", "text": text})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": _ASTRA_SCHEMA}},
            messages=[{"role": "user", "content": content}],
        )
        if response.stop_reason == "refusal":
            return Decision(False, f"model declined to answer ({response.stop_details})", raw=None)
        out = json.loads(next(b.text for b in response.content if b.type == "text"))
        return Decision(bool(out["intervene"]), f"{out['failure_mode']}: {out['reason']}", raw=out)


def make_monitor(kind: str, patience: int = 20) -> Monitor:
    if kind == "stall":
        return StallMonitor(patience)
    if kind == "astra":
        return AstraMonitor()
    if kind == "claude":
        return ClaudeMonitor()
    raise ValueError(f"unknown monitor {kind!r}")
