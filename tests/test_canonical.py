"""Property tests for the canonical form and the equivalence check."""

from __future__ import annotations

import math

import sympy
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from finqa_search.dsl import (
    COMMUTATIVE,
    Num,
    Program,
    ProgramError,
    Ref,
    Step,
    canonical,
    equivalent,
    execute,
    parse_program,
    value_key,
)

LEAVES = ("2", "3", "5", "7")
ARITH = ("add", "subtract", "multiply", "divide")


@st.composite
def programs(draw: st.DrawFn, max_steps: int = 4, leaves: tuple[str, ...] = LEAVES) -> Program:
    n = draw(st.integers(1, max_steps))
    steps: list[Step] = []
    for i in range(n):
        op = draw(st.sampled_from(ARITH))
        args = []
        for _ in range(2):
            if i and draw(st.booleans()):
                args.append(Ref(draw(st.integers(0, i - 1))))
            else:
                text = draw(st.sampled_from(leaves))
                args.append(Num(text, float(text)))
        steps.append(Step(op, args[0], args[1]))
    return Program(tuple(steps))


def swap_commutative(program: Program, mask: list[bool]) -> Program:
    steps = []
    for step, flip in zip(program.steps, mask, strict=False):
        if flip and step.op in COMMUTATIVE:
            steps.append(Step(step.op, step.b, step.a))
        else:
            steps.append(step)
    steps.extend(program.steps[len(steps) :])
    return Program(tuple(steps))


def to_sympy(program: Program, symbols: dict[str, sympy.Symbol]) -> sympy.Expr:
    """The program as a sympy expression; ``symbols`` maps leaf values to shared symbols."""
    results: list[sympy.Expr] = []
    for step in program.steps:
        xs = []
        for arg in (step.a, step.b):
            if isinstance(arg, Ref):
                xs.append(results[arg.step])
            else:
                assert isinstance(arg, Num)
                key = value_key(arg.value)
                xs.append(symbols.setdefault(key, sympy.Symbol(f"v{len(symbols)}")))
        x, y = xs
        if step.op == "add":
            results.append(x + y)
        elif step.op == "subtract":
            results.append(x - y)
        elif step.op == "multiply":
            results.append(x * y)
        else:
            results.append(x / y)
    return results[-1]


def run(program: Program) -> float | None:
    try:
        value = execute(program, [])
    except ProgramError:
        return None
    return float(value)


@given(programs(), st.lists(st.booleans(), min_size=4, max_size=4))
def test_canonical_form_is_invariant_under_commutation(program: Program, mask: list[bool]) -> None:
    swapped = swap_commutative(program, mask)
    assert canonical(swapped) == canonical(program)
    assert equivalent(swapped, program)


@given(programs())
def test_canonical_form_survives_a_render_and_parse(program: Program) -> None:
    again = parse_program(program.render())
    assert again == program
    assert canonical(again) == canonical(program)


@given(programs(), programs())
def test_equal_canonical_forms_execute_to_the_same_value(p: Program, q: Program) -> None:
    assume(canonical(p) == canonical(q))
    vp, vq = run(p), run(q)
    assume(vp is not None and vq is not None)
    assert math.isclose(vp, vq, rel_tol=1e-9, abs_tol=1e-12)


@settings(max_examples=300, deadline=None)
@given(programs(max_steps=3, leaves=("2", "3", "5")), programs(max_steps=3, leaves=("2", "3", "5")))
def test_equivalence_agrees_with_sympy(p: Program, q: Program) -> None:
    symbols: dict[str, sympy.Symbol] = {}
    ep, eq = to_sympy(p, symbols), to_sympy(q, symbols)
    assume(not (ep.has(sympy.zoo, sympy.nan) or eq.has(sympy.zoo, sympy.nan)))
    diff = sympy.simplify(ep - eq)
    assume(not diff.has(sympy.zoo, sympy.nan))
    assert equivalent(p, q) == (diff == 0)


@given(programs())
def test_equivalence_is_reflexive_and_seed_independent(program: Program) -> None:
    assert equivalent(program, program, seed=0)
    assert equivalent(program, program, seed=12345)
