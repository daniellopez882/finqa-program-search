"""Execute gold programs against their recorded answers.

The interpreter is checked against the whole training split: every gold program must
parse, run and reproduce ``exe_ans``. The few that do not are annotation noise; each one
is listed with a reason in ``fixtures/gold_allowlist.json`` after being read by hand.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from finqa_search.dataset import Example
from finqa_search.dsl import ProgramError, answers_match, execute, parse_program


def load_allowlist(path: Path | None = None) -> dict[str, str]:
    if path is not None:
        text = path.read_text(encoding="utf-8")
    else:
        ref = resources.files("finqa_search").joinpath("fixtures/gold_allowlist.json")
        text = ref.read_text("utf-8")
    return dict(json.loads(text))


def check_example(example: Example) -> str | None:
    """None if the gold program reproduces the answer, else why not."""
    try:
        program = parse_program(example.program)
    except ProgramError as exc:
        return f"parse: {exc}"
    try:
        value = execute(program, example.table)
    except ProgramError as exc:
        return f"execute: {exc}"
    if not answers_match(value, example.exe_ans):
        return f"answer: got {value!r}, recorded {example.exe_ans!r}"
    return None


@dataclass
class GoldReport:
    checked: int = 0
    passed: int = 0
    allowed: dict[str, str] = field(default_factory=dict)
    unexpected: dict[str, str] = field(default_factory=dict)
    stale: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "checked": self.checked,
            "reproduced": self.passed,
            "allowlisted": len(self.allowed),
            "unexpected_failures": len(self.unexpected),
            "allowlisted_but_passing": self.stale,
        }


def check_split(examples: Sequence[Example], allowlist: dict[str, str]) -> GoldReport:
    report = GoldReport()
    for ex in examples:
        report.checked += 1
        problem = check_example(ex)
        if problem is None:
            report.passed += 1
            if ex.id in allowlist:
                report.stale.append(ex.id)
        elif ex.id in allowlist:
            report.allowed[ex.id] = problem
        else:
            report.unexpected[ex.id] = problem
    return report
