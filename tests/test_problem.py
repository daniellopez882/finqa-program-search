"""ProgramProblem: memoised expansions, derivation log-likelihoods and the answer rule."""

from __future__ import annotations

import math

import numpy as np
import pytest

from finqa_search.prior import UniformPrior
from finqa_search.search.baselines import beam, exhaustive_depth2, greedy
from finqa_search.search.problem import LOG_PRIOR_FLOOR, ProgramProblem
from finqa_search.solver import values_contain
from finqa_search.space import SUB, ActionSpace


class LengthRanker:
    """Scores a program by its first-step operation, so tests know the ranking."""

    def score(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(X[:, 1] - 5.0)))  # op_first column


def leaf(space: ActionSpace, text: str) -> int:
    return next(i for i, lf in enumerate(space.ctx.leaves) if lf.arg.render() == text)


def test_expansions_are_memoised(space: ActionSpace) -> None:
    problem = ProgramProblem(space, UniformPrior(), None)
    root = problem.root()
    a = problem.expand([root, root])
    b = problem.expand([root])
    assert a[0] is a[1] is b[0]
    assert problem.counters.expansions == 1
    assert np.isclose(a[0].priors.sum(), 1.0)


def test_child_tracks_the_best_derivation_log_likelihood(space: ActionSpace) -> None:
    problem = ProgramProblem(space, UniformPrior(), None)
    root = problem.root()
    (exp,) = problem.expand([root])
    x, y = leaf(space, "5829"), leaf(space, "5735")
    child = problem.child(root, (SUB, x, y))
    assert problem.log_prior[child.key] == pytest.approx(math.log(1.0 / len(exp)))
    (exp2,) = problem.expand([child])
    grand = problem.child(child, exp2.actions[0])
    expected = math.log(1.0 / len(exp)) + math.log(exp2.priors[0])
    assert problem.log_prior[grand.key] == pytest.approx(expected)


def test_unexpanded_parents_leave_the_log_likelihood_unknown(space: ActionSpace) -> None:
    problem = ProgramProblem(space, UniformPrior(), None)
    x, y = leaf(space, "5829"), leaf(space, "5735")
    child = problem.child(problem.root(), (SUB, x, y))  # root never expanded
    assert child.key not in problem.log_prior


def test_selection_adds_weighted_log_prior_to_log_ranker(space: ActionSpace) -> None:
    problem = ProgramProblem(space, UniformPrior(), LengthRanker(), prior_weight=0.5)  # type: ignore[arg-type]
    root = problem.root()
    problem.expand([root])
    x, y = leaf(space, "5829"), leaf(space, "5735")
    child = problem.child(root, (SUB, x, y))
    (r,) = problem.stop_values([child])
    lp = problem.log_prior[child.key]
    assert problem.selection_score(child) == pytest.approx(math.log(r) + 0.5 * lp)
    best = problem.best()
    assert best is not None
    assert best[0] == child


def test_programs_with_unknown_derivations_get_the_floor(space: ActionSpace) -> None:
    problem = ProgramProblem(space, UniformPrior(), LengthRanker(), prior_weight=1.0)  # type: ignore[arg-type]
    x, y = leaf(space, "5829"), leaf(space, "5735")
    child = space.child(space.root(), (SUB, x, y))  # built outside the problem
    (r,) = problem.stop_values([child])
    assert problem.selection_score(child) == pytest.approx(math.log(r) + LOG_PRIOR_FLOOR)


def test_baselines_return_their_candidates(space: ActionSpace) -> None:
    problem = ProgramProblem(space, UniformPrior(), LengthRanker())  # type: ignore[arg-type]
    g = greedy(problem)
    assert g.state is not None
    assert g.candidates == [g.state]
    b = beam(problem, 3, rerank=True)
    assert b.state in b.candidates
    assert b.score == max(problem.selection_score(s) for s in b.candidates)
    d = exhaustive_depth2(problem)
    assert d.candidate_values is not None
    assert d.candidate_bool is not None
    assert len(d.candidate_values) == d.finished
    gold = 94.0 / 5735.0  # (5829 - 5735) / 5735 is a two-step program over these leaves
    assert values_contain(d.candidate_values, d.candidate_bool, gold)
    assert not values_contain(d.candidate_values, d.candidate_bool, 123456.789)
