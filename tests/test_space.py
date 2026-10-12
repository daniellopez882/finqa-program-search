"""The typed program space: leaves, legal actions, transposition keys and gold encoding."""

from __future__ import annotations

import numpy as np
import pytest

from finqa_search.context import CONSTANT_SLOTS, QuestionContext
from finqa_search.dataset import Example
from finqa_search.dsl import Const, Num, answers_match, execute, parse_program
from finqa_search.features import ACTION_FEATURES, OPS
from finqa_search.space import ADD, DIV, EXP, GT, MUL, STOP_ACTION, SUB, ActionSpace
from finqa_search.training import find_action, gold_trajectory


def leaf_of(space: ActionSpace, text: str) -> int:
    return next(i for i, leaf in enumerate(space.ctx.leaves) if leaf.arg.render() == text)


def test_context_reads_numbers_from_the_top_facts_then_constants(ctx: QuestionContext) -> None:
    numbers = [leaf.arg.render() for leaf in ctx.leaves if isinstance(leaf.arg, Num)]
    consts = [leaf.arg.render() for leaf in ctx.leaves if isinstance(leaf.arg, Const)]
    assert {"5829", "5735"} <= set(numbers)
    assert consts
    assert set(consts) <= set(CONSTANT_SLOTS)
    first_const = next(i for i, leaf in enumerate(ctx.leaves) if isinstance(leaf.arg, Const))
    assert all(isinstance(leaf.arg, Const) for leaf in ctx.leaves[first_const:])
    assert ctx.leaf_feats.shape[0] == ctx.n_leaves
    assert "net revenue" in [row.label for row in ctx.rows]


def test_root_actions_respect_the_typing_rules(space: ActionSpace) -> None:
    batch = space.legal(space.root())
    assert batch.features.shape == (len(batch), len(ACTION_FEATURES))
    L = space.L
    for op, a, b in batch.actions:
        assert op != EXP  # exp needs a step result as its base
        assert (op, a, b) != STOP_ACTION
        if OPS[op].startswith("table_"):
            assert b == -1
            continue
        assert a < L
        assert b < L
        names = [space._const_name_of(int(i)) for i in (a, b)]
        assert not all(names)  # two constants never meet
        for slot, name in enumerate(names):
            if name:
                assert (OPS[op], slot) in CONSTANT_SLOTS[name]


def test_commutative_ops_appear_in_one_argument_order(space: ActionSpace) -> None:
    acts = {tuple(int(v) for v in a) for a in space.legal(space.root()).actions}
    for op in (ADD, MUL):
        for _, a, b in (x for x in acts if x[0] == op):
            assert (op, b, a) not in acts
    x, y = leaf_of(space, "5829"), leaf_of(space, "5735")
    assert (SUB, x, y) in acts
    assert (SUB, y, x) in acts


def test_results_match_the_interpreter(space: ActionSpace) -> None:
    root = space.root()
    batch = space.legal(root)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(batch), size=40, replace=False):
        act = batch.action(int(i))
        child = space.child(root, act)
        program = space.program(child)
        value = execute(program, space.ctx.table)
        if isinstance(value, bool):
            assert batch.results[i] == float(value)
        else:
            assert value == pytest.approx(batch.results[i])
            assert answers_match(child.answer, value)


def test_stop_only_when_every_step_is_used(space: ActionSpace) -> None:
    root = space.root()
    x, y = leaf_of(space, "5829"), leaf_of(space, "5735")
    s1 = space.child(root, (SUB, x, y))
    assert s1.can_stop
    s2 = space.child(s1, (ADD, x, y))  # a second leaf-only step leaves #0 dangling
    assert not s2.can_stop
    legal = space.legal(s2)
    assert find_action(legal, STOP_ACTION) is None
    # Every legal action from here must consume a dangling result, as only 2 steps remain.
    L = space.L
    for _op, a, b in legal.actions:
        assert a >= L or b >= L


def test_greater_must_be_the_last_step(space: ActionSpace) -> None:
    x, y = leaf_of(space, "5829"), leaf_of(space, "5735")
    s = space.child(space.root(), (GT, x, y))
    assert s.answer is True
    legal = space.legal(s)
    assert [tuple(a) for a in legal.actions] == [STOP_ACTION]


def test_transposed_programs_share_a_key(space: ActionSpace) -> None:
    x, y = leaf_of(space, "5829"), leaf_of(space, "5735")
    z = leaf_of(space, "5516")
    root = space.root()
    a = space.child(space.child(root, (SUB, x, y)), (SUB, x, z))
    b = space.child(space.child(root, (SUB, x, z)), (SUB, x, y))
    # Same steps, different last step: the answer so far differs, so the keys do too.
    assert a.key != b.key
    L = space.L
    c = space.child(a, (DIV, L, L + 1))
    d = space.child(b, (DIV, L + 1, L))
    assert c.key == d.key
    assert c.answer == pytest.approx(d.answer)


def test_gold_program_encodes_to_a_legal_trajectory(space: ActionSpace, example: Example) -> None:
    traj = gold_trajectory(space, example)
    assert traj is not None
    assert traj[-1] == STOP_ACTION
    state = space.root()
    for act in traj[:-1]:
        state = space.child(state, act)
    assert space.program(state).render() == example.program
    assert answers_match(state.answer, example.exe_ans)


def test_unreachable_gold_programs_are_reported(space: ActionSpace, example: Example) -> None:
    other = example.model_copy(update={"program": "subtract(9999, 5735)"})
    assert gold_trajectory(space, other) is None
    assert space.encode(parse_program("table_sum(gross margin, none)")) is None


def test_max_steps_bounds_the_depth(ctx: QuestionContext) -> None:
    space = ActionSpace(ctx, max_steps=1)
    x = next(i for i, leaf in enumerate(ctx.leaves) if leaf.arg.render() == "5829")
    y = next(i for i, leaf in enumerate(ctx.leaves) if leaf.arg.render() == "5735")
    s = space.child(space.root(), (SUB, x, y))
    assert [tuple(a) for a in space.legal(s).actions] == [STOP_ACTION]
