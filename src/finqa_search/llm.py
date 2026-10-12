"""An LLM as an optional expansion prior, behind a small provider interface.

``OpenAICompatibleProvider`` speaks the ``/chat/completions`` protocol shared by OpenAI,
Ollama, vLLM and LM Studio. ``ScriptedProvider`` is deterministic and offline; tests and
offline runs use it. ``LLMPrior`` asks the model for candidate next steps in the program
language, keeps the ones that are legal in the current state and mixes their frequency
into a base prior. No benchmark in this repository runs it against a real model.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from finqa_search.dsl import ProgramError, parse_program
from finqa_search.prior import Prior
from finqa_search.space import STOP_ACTION, Action, ActionBatch, ActionSpace, State


class Provider(Protocol):
    def complete(self, messages: list[dict[str, str]], *, n: int, temperature: float) -> list[str]:
        """``n`` completions of a chat conversation."""
        ...


@dataclass
class OpenAICompatibleProvider:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 60.0
    max_tokens: int = 64

    def complete(self, messages: list[dict[str, str]], *, n: int, temperature: float) -> list[str]:
        body = {
            "model": self.model,
            "messages": messages,
            "n": n,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return [c["message"]["content"] or "" for c in payload.get("choices", [])]


@dataclass
class ScriptedProvider:
    """Answers from a function of the prompt; records every call."""

    respond: Callable[[list[dict[str, str]], int], list[str]]
    calls: list[list[dict[str, str]]] = field(default_factory=list)

    def complete(self, messages: list[dict[str, str]], *, n: int, temperature: float) -> list[str]:
        self.calls.append(messages)
        return self.respond(messages, n)


SYSTEM_PROMPT = (
    "You write the next step of a FinQA program. Operations: add, subtract, multiply, "
    "divide, exp, greater, table_sum, table_average, table_max, table_min. Arguments are "
    "numbers copied from the evidence, constants such as const_100, earlier results #0, #1, "
    "or a table row label followed by none. Reply with exactly one step, e.g. "
    "subtract(5829, 5735), or the word stop if the program is finished."
)

_STEP_RE = re.compile(r"[a-z_]+\([^()]*(?:\([^()]*\)[^()]*)*\)|\bstop\b")


def build_messages(space: ActionSpace, state: State) -> list[dict[str, str]]:
    ctx = space.ctx
    numbers = ", ".join(leaf.arg.render() for leaf in ctx.leaves)
    rows = "; ".join(row.label for row in ctx.rows) or "(none)"
    so_far = space.program(state).render() if state.steps else "(empty)"
    user = (
        f"Question: {ctx.question}\n"
        f"Numbers you may use: {numbers}\n"
        f"Table rows: {rows}\n"
        f"Program so far: {so_far}\n"
        "Next step:"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_proposal(space: ActionSpace, state: State, text: str) -> Action | None:
    """Map one completion onto a legal action of ``state`` (None if it is not one)."""
    m = _STEP_RE.search(text.strip().lower())
    if not m:
        return None
    if m.group(0) == "stop":
        return STOP_ACTION if state.can_stop else None
    prefix = space.program(state).render() if state.steps else ""
    try:
        program = parse_program(f"{prefix}, {m.group(0)}" if prefix else m.group(0))
    except ProgramError:
        return None
    acts = space.encode(program)
    if acts is None:
        return None
    action = acts[-1]
    batch = space.legal(state)
    hit = (batch.actions == np.array(action)).all(axis=1)
    return action if hit.any() else None


@dataclass
class LLMPrior:
    """``(1 - weight) * base + weight * share of the n samples proposing each action``."""

    provider: Provider
    base: Prior
    samples: int = 8
    temperature: float = 0.7
    weight: float = 0.5
    rejected: int = 0

    def __call__(
        self, space: ActionSpace, states: list[State], batches: list[ActionBatch]
    ) -> list[np.ndarray]:
        base = self.base(space, states, batches)
        out: list[np.ndarray] = []
        for state, batch, p in zip(states, batches, base, strict=True):
            texts = self.provider.complete(
                build_messages(space, state), n=self.samples, temperature=self.temperature
            )
            votes: Counter[Action] = Counter()
            for t in texts:
                act = parse_proposal(space, state, t)
                if act is None:
                    self.rejected += 1
                else:
                    votes[act] += 1
            if not votes:
                out.append(p)
                continue
            llm = np.zeros(len(batch))
            for act, count in votes.items():
                llm[(batch.actions == np.array(act)).all(axis=1)] = count
            llm /= llm.sum()
            out.append((1 - self.weight) * p + self.weight * llm)
        return out


def scripted_from_steps(steps: Sequence[str]) -> ScriptedProvider:
    """A provider that walks through ``steps`` one call at a time (then says stop)."""
    it = iter(steps)

    def respond(_: list[dict[str, str]], n: int) -> list[str]:
        nxt = next(it, "stop")
        return [nxt] * n

    return ScriptedProvider(respond)
