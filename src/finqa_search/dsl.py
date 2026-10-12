"""The FinQA program language: parser, interpreter, canonical form and equivalence.

A program is a sequence of steps ``op(arg1, arg2)``. Arguments are numbers as written in
the report (``5829``, ``3.8%``), named constants (``const_100``, ``const_m1``), references
to earlier step results (``#0``) or, for the four table operations, a table row label
followed by ``none``. Every step result can be referenced later; the last step is the
answer. Number semantics follow the FinQA release: commas are ignored and ``x%`` means
``x / 100``.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Final

ARITH_OPS: Final = ("add", "subtract", "multiply", "divide", "exp", "greater")
TABLE_OPS: Final = ("table_sum", "table_average", "table_max", "table_min")
ALL_OPS: Final = ARITH_OPS + TABLE_OPS
COMMUTATIVE: Final = frozenset({"add", "multiply"})

CONSTANTS: Final[dict[str, float]] = {
    "const_m1": -1.0,
    "const_1": 1.0,
    "const_2": 2.0,
    "const_3": 3.0,
    "const_4": 4.0,
    "const_5": 5.0,
    "const_6": 6.0,
    "const_7": 7.0,
    "const_8": 8.0,
    "const_9": 9.0,
    "const_10": 10.0,
    "const_12": 12.0,
    "const_100": 100.0,
    "const_1000": 1000.0,
    "const_10000": 10000.0,
    "const_100000": 100000.0,
    "const_1000000": 1000000.0,
    "const_10000000": 10000000.0,
    "const_1000000000": 1000000000.0,
}

Answer = float | bool


class ProgramError(ValueError):
    """The program text does not parse, or a step cannot be executed."""


@dataclass(frozen=True, slots=True)
class Num:
    """A literal number. ``value`` already applies the percent rule."""

    text: str
    value: float

    def render(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class Const:
    name: str

    @property
    def value(self) -> float:
        return CONSTANTS[self.name]

    def render(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Ref:
    step: int

    def render(self) -> str:
        return f"#{self.step}"


@dataclass(frozen=True, slots=True)
class Row:
    """A table row label, used by the table operations."""

    label: str

    def render(self) -> str:
        return self.label


@dataclass(frozen=True, slots=True)
class NoArg:
    """The ``none`` placeholder in the second slot of a table operation."""

    def render(self) -> str:
        return "none"


Arg = Num | Const | Ref | Row | NoArg


@dataclass(frozen=True, slots=True)
class Step:
    op: str
    a: Arg
    b: Arg

    def render(self) -> str:
        return f"{self.op}({self.a.render()}, {self.b.render()})"


@dataclass(frozen=True, slots=True)
class Program:
    steps: tuple[Step, ...]

    def render(self) -> str:
        return ", ".join(s.render() for s in self.steps)

    def __len__(self) -> int:
        return len(self.steps)


_NUMBER_RE = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)$")


def number_value(text: str) -> float | None:
    """FinQA's string-to-number rule: drop commas, ``%`` divides by 100, else None."""
    t = text.replace(",", "").strip()
    pct = t.endswith("%")
    if pct:
        t = t[:-1].strip()
    if t.startswith("$"):
        t = t[1:].strip()
    if not _NUMBER_RE.match(t):
        return None
    v = float(t)
    return v / 100.0 if pct else v


def parse_arg(token: str, op: str, position: int) -> Arg:
    tok = token.strip()
    if op in TABLE_OPS:
        if position == 1:
            if tok != "none":
                raise ProgramError(f"{op} takes 'none' as its second argument, got {tok!r}")
            return NoArg()
        if tok.startswith("#") and tok[1:].isdigit():
            return Ref(int(tok[1:]))
        return Row(tok)
    if tok.startswith("#") and tok[1:].isdigit():
        return Ref(int(tok[1:]))
    if tok in CONSTANTS:
        return Const(tok)
    value = number_value(tok)
    if value is None:
        raise ProgramError(f"not a number, constant or reference: {tok!r}")
    return Num(tok, value)


def _split_steps(text: str) -> list[tuple[str, str]]:
    """Split ``op(args), op(args)`` into (op, args) pairs, allowing parentheses in labels."""
    out: list[tuple[str, str]] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " ,\t\n":
            i += 1
        if i >= n:
            break
        j = text.find("(", i)
        if j < 0:
            raise ProgramError(f"expected '(' after position {i} in {text!r}")
        op = text[i:j].strip()
        depth, k = 1, j + 1
        while k < n and depth:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        if depth:
            raise ProgramError(f"unbalanced parentheses in {text!r}")
        out.append((op, text[j + 1 : k - 1]))
        i = k
    return out


def parse_program(text: str) -> Program:
    """Parse FinQA program text such as ``subtract(5829, 5735), divide(#0, 5735)``."""
    raw_steps = _split_steps(text.strip().removesuffix("EOF").strip())
    if not raw_steps:
        raise ProgramError("empty program")
    steps: list[Step] = []
    for index, (op, args) in enumerate(raw_steps):
        if op not in ALL_OPS:
            raise ProgramError(f"unknown operation {op!r}")
        # Row labels may contain commas, so table operations split on the last comma.
        head, sep, tail = args.rpartition(",") if op in TABLE_OPS else args.partition(",")
        if not sep:
            raise ProgramError(f"{op} needs two arguments, got {args!r}")
        a, b = parse_arg(head, op, 0), parse_arg(tail, op, 1)
        for arg in (a, b):
            if isinstance(arg, Ref) and arg.step >= index:
                raise ProgramError(
                    f"step {index} references #{arg.step}, which is not computed yet"
                )
        steps.append(Step(op, a, b))
    return Program(tuple(steps))


def _cell_value(cell: str) -> float | None:
    """A table cell as FinQA's interpreter reads it: '$' dropped, text after '(' ignored."""
    return number_value(cell.replace("$", "").split("(")[0].strip())


def table_row_values(table: list[list[str]], label: str) -> list[float]:
    """The numeric cells of the row labelled ``label``.

    When a label repeats (one block per year, say), the last row wins, as in the FinQA
    release's own interpreter, which builds a label -> row dict.
    """
    for row in reversed(table):
        if row and row[0] == label:
            values = [_cell_value(c) for c in row[1:]]
            if not values or any(v is None for v in values):
                raise ProgramError(f"row {label!r} has a non-numeric cell")
            return [v for v in values if v is not None]
    raise ProgramError(f"no table row labelled {label!r}")


def apply_op(op: str, x: float, y: float) -> Answer:
    if op == "add":
        r = x + y
    elif op == "subtract":
        r = x - y
    elif op == "multiply":
        r = x * y
    elif op == "divide":
        if y == 0:
            raise ProgramError("division by zero")
        r = x / y
    elif op == "exp":
        try:
            r = x**y
        except (OverflowError, ZeroDivisionError) as exc:
            raise ProgramError(f"exp({x}, {y}) is undefined") from exc
        if isinstance(r, complex):
            raise ProgramError(f"exp({x}, {y}) is complex")
    elif op == "greater":
        return x > y
    else:
        raise ProgramError(f"{op} is not an arithmetic operation")
    if not math.isfinite(r):
        raise ProgramError(f"{op}({x}, {y}) is not finite")
    return r


def apply_table_op(op: str, values: list[float]) -> float:
    if op == "table_sum":
        return math.fsum(values)
    if op == "table_average":
        return math.fsum(values) / len(values)
    if op == "table_max":
        return max(values)
    if op == "table_min":
        return min(values)
    raise ProgramError(f"{op} is not a table operation")


def _arg_number(arg: Arg, results: list[Answer]) -> float:
    if isinstance(arg, Num | Const):
        return arg.value
    if isinstance(arg, Ref):
        r = results[arg.step]
        if isinstance(r, bool):
            raise ProgramError("a yes/no result cannot be used in arithmetic")
        return r
    raise ProgramError(f"{arg!r} is not numeric")


def execute_steps(program: Program, table: list[list[str]]) -> list[Answer]:
    """Run every step and return all intermediate results."""
    results: list[Answer] = []
    for step in program.steps:
        if step.op in TABLE_OPS:
            if not isinstance(step.a, Row):
                raise ProgramError(f"{step.op} needs a row label")
            results.append(apply_table_op(step.op, table_row_values(table, step.a.label)))
        else:
            x = _arg_number(step.a, results)
            y = _arg_number(step.b, results)
            results.append(apply_op(step.op, x, y))
    return results


def execute(program: Program, table: list[list[str]]) -> Answer:
    return execute_steps(program, table)[-1]


def round_answer(value: Answer) -> Answer:
    return value if isinstance(value, bool) else round(value, 5)


def answers_match(pred: Answer | None, gold: float | str) -> bool:
    """FinQA execution accuracy: yes/no compared as words, numbers after rounding to 5 dp."""
    if pred is None:
        return False
    if isinstance(gold, str):
        g = gold.strip().lower()
        if g in ("yes", "no"):
            return isinstance(pred, bool) and (pred == (g == "yes"))
        gv = number_value(g)
        if gv is None:
            return False
        gold = gv
    if isinstance(pred, bool):
        return False
    return round(pred, 5) == round(float(gold), 5)


# Canonical form -----------------------------------------------------------------------


def value_key(value: float) -> str:
    """A stable text key for a leaf value, so 5,829 and 5829 are the same leaf."""
    return repr(float(round(value, 9)))


def _canon_leaf(arg: Arg) -> str:
    if isinstance(arg, Num | Const):
        return value_key(arg.value)
    if isinstance(arg, Row):
        return f"row:{arg.label}"
    return "none"


def canonical_steps(program: Program) -> list[str]:
    """Canonical expression text for every step, with references inlined.

    Arguments of add and multiply are sorted, so ``add(a, b)`` and ``add(b, a)`` share a
    form, and so do programs that compute the same sub-expressions in another order.
    """
    forms: list[str] = []
    for step in program.steps:
        parts: list[str] = []
        for arg in (step.a, step.b):
            if isinstance(arg, Ref):
                parts.append(forms[arg.step])
            else:
                parts.append(_canon_leaf(arg))
        if step.op in COMMUTATIVE:
            parts.sort()
        forms.append(f"{step.op}({parts[0]},{parts[1]})")
    return forms


def canonical(program: Program) -> str:
    return canonical_steps(program)[-1]


# Equivalence --------------------------------------------------------------------------

_PRIME: Final = (1 << 61) - 1


def _modexpr(program: Program, assignment: dict[str, int], rng: random.Random) -> list[int | None]:
    """Evaluate every step modulo a prime with leaves replaced by random field elements.

    ``exp`` and ``greater`` are not rational functions; they become opaque atoms keyed by
    the canonical form of their arguments, so they match only structurally.
    """
    forms = canonical_steps(program)
    vals: list[int | None] = []

    def leaf(key: str) -> int:
        if key not in assignment:
            assignment[key] = rng.randrange(2, _PRIME - 1)
        return assignment[key]

    for index, step in enumerate(program.steps):
        if step.op in TABLE_OPS or step.op in ("exp", "greater"):
            vals.append(leaf(f"atom:{forms[index]}"))
            continue
        x, y = (
            vals[arg.step] if isinstance(arg, Ref) else leaf(_canon_leaf(arg))
            for arg in (step.a, step.b)
        )
        if x is None or y is None:
            vals.append(None)
        elif step.op == "add":
            vals.append((x + y) % _PRIME)
        elif step.op == "subtract":
            vals.append((x - y) % _PRIME)
        elif step.op == "multiply":
            vals.append((x * y) % _PRIME)
        else:  # divide
            vals.append(None if y == 0 else (x * pow(y, _PRIME - 2, _PRIME)) % _PRIME)
    return vals


def equivalent(p: Program, q: Program, trials: int = 3, seed: int = 0) -> bool:
    """Whether two programs compute the same expression of their leaf values.

    Leaves are identified by value (constants included), and both programs are evaluated
    at random points of a prime field: two different rational functions agree at a random
    point with probability at most degree / 2^61 (Schwartz-Zippel), so a few trials decide
    equivalence for programs of FinQA's size. The official FinQA script asks sympy the
    same question; ``tests/test_equivalence.py`` checks that the two agree.
    """
    rng = random.Random(seed)
    for _ in range(trials):
        assignment: dict[str, int] = {}
        a = _modexpr(p, assignment, rng)[-1]
        b = _modexpr(q, assignment, rng)[-1]
        if a is None or b is None:
            return canonical(p) == canonical(q)
        if a != b:
            return False
    return True
