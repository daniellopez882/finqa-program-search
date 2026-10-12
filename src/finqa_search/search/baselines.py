"""Greedy decoding, beam search and exhaustive depth-2 enumeration over the same space."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from finqa_search.ranker import program_features
from finqa_search.search.problem import ProgramProblem, as_action
from finqa_search.space import ADD, DIV, EXP, GT, MUL, STOP_ACTION, SUB, State, safe_apply


@dataclass
class BaselineOutcome:
    """The chosen program plus every finished program the method looked at.

    ``candidates`` lists them as states; the exhaustive baseline scores too many for
    that and reports their final values instead (``candidate_values``, with
    ``candidate_bool`` marking yes/no results).
    """

    state: State | None
    score: float | None
    seconds: float
    finished: int
    candidates: list[State] = field(default_factory=list)
    candidate_values: np.ndarray | None = None
    candidate_bool: np.ndarray | None = None


def greedy(problem: ProgramProblem) -> BaselineOutcome:
    """Follow the most probable action until stop is the most probable action."""
    start = time.perf_counter()
    state = problem.root()
    while True:
        exp = problem.expand([state])[0]
        if not len(exp):
            return BaselineOutcome(None, None, time.perf_counter() - start, 0)
        act = exp.actions[0]
        if act == STOP_ACTION:
            break
        state = problem.child(state, act)
    return BaselineOutcome(state, None, time.perf_counter() - start, 1, [state])


def beam_finished(problem: ProgramProblem, width: int) -> list[tuple[float, State]]:
    """Every program that finished inside a beam of ``width`` on summed log prior."""
    beams: list[tuple[float, State]] = [(0.0, problem.root())]
    finished: list[tuple[float, State]] = []
    while beams:
        exps = problem.expand([s for _, s in beams])
        cands: list[tuple[float, State, tuple[int, int, int]]] = []
        for (lp, state), exp in zip(beams, exps, strict=True):
            cands.extend(
                (lp + math.log(max(exp.priors[i], 1e-12)), state, as_action(exp.actions[i]))
                for i in range(min(width, len(exp)))
            )
        cands.sort(key=lambda c: -c[0])
        nxt: dict[str, tuple[float, State]] = {}
        taken = 0
        for lp, state, act in cands:
            if taken >= width:
                break
            if act == STOP_ACTION:
                finished.append((lp, state))
                taken += 1
                continue
            child = problem.child(state, act)
            if child.key in nxt:
                continue
            nxt[child.key] = (lp, child)
            taken += 1
        beams = list(nxt.values())
    return finished


def beam(problem: ProgramProblem, width: int, rerank: bool) -> BaselineOutcome:
    """Beam search on summed log prior; the answer is the best finished program by log
    prior, or, when ``rerank`` is set, by the same ranker-plus-prior score MCTS uses."""
    start = time.perf_counter()
    finished = beam_finished(problem, width)
    if not finished:
        return BaselineOutcome(None, None, time.perf_counter() - start, 0)
    states = [s for _, s in finished]
    if rerank:
        problem.stop_values(states)
        scores = [problem.selection_score(s) for s in states]
        i = int(np.argmax(scores))
        return BaselineOutcome(
            finished[i][1], scores[i], time.perf_counter() - start, len(finished), states
        )
    best = max(finished, key=lambda f: f[0])
    return BaselineOutcome(best[1], None, time.perf_counter() - start, len(finished), states)


def exhaustive_depth2(problem: ProgramProblem, chunk: int = 60_000) -> BaselineOutcome:
    """Score every one-step program and every two-step program whose second step consumes
    the first, with the ranker alone, and return the best. No prior is used."""
    start = time.perf_counter()
    if problem.ranker is None:
        raise ValueError("the exhaustive baseline needs a ranker")
    ranker = problem.ranker
    space = problem.space
    ctx = space.ctx
    L = space.L
    first = space.static_actions
    v1 = space.static_results
    is_gt = first[:, 0] == GT
    # Second steps that use #0 (index L) once, typed as in the space.
    refs = [L]
    leaves = list(range(L))
    second: list[tuple[int, int, int]] = []
    for op in (ADD, SUB, MUL, DIV, EXP, GT):
        second.extend(
            (op, a, b)
            for a, b in space.typed_pairs(op, refs, leaves) + space.typed_pairs(op, leaves, refs)
        )
    sec = np.array(second, dtype=np.int64).reshape(-1, 3)

    n1, n2 = len(first), len(sec)
    best_score, best_prog = -1.0, None
    total = 0
    # One-step programs.
    steps1 = first[:, None, :].copy()
    vals1 = v1[:, None].copy()
    X = program_features(ctx, steps1, vals1, np.ones(n1, dtype=np.int64))
    s1 = ranker.score(X)
    total += n1
    finals: list[np.ndarray] = [v1]
    final_bool: list[np.ndarray] = [is_gt]
    if n1:
        i = int(np.argmax(s1))
        best_score, best_prog = float(s1[i]), ([tuple(first[i])], [v1[i]])

    # Two-step programs, in chunks of first steps.
    usable = np.flatnonzero(~is_gt)
    per = max(1, chunk // max(n2, 1))
    for lo in range(0, len(usable), per):
        idx = usable[lo : lo + per]
        f = first[idx]
        fv = v1[idx]
        a_is_ref = sec[:, 1] == L
        other = np.where(a_is_ref, sec[:, 2], sec[:, 1])
        ov = ctx.leaf_values[other]
        xa = np.where(a_is_ref[None, :], fv[:, None], ov[None, :])
        xb = np.where(a_is_ref[None, :], ov[None, :], fv[:, None])
        res = np.full(xa.shape, np.nan)
        for op in (ADD, SUB, MUL, DIV, EXP, GT):
            m = sec[:, 0] == op
            if m.any():
                res[:, m] = safe_apply(op, xa[:, m], xb[:, m])
        ok = np.isfinite(res)
        ok &= ~(np.isin(sec[:, 0], (SUB, DIV, GT))[None, :] & (xa == xb))
        rows, cols = np.nonzero(ok)
        if not len(rows):
            continue
        steps = np.zeros((len(rows), 2, 3), dtype=np.int64)
        steps[:, 0] = f[rows]
        steps[:, 1] = sec[cols]
        vals = np.stack([fv[rows], res[rows, cols]], axis=1)
        X = program_features(ctx, steps, vals, np.full(len(rows), 2, dtype=np.int64))
        s = ranker.score(X)
        total += len(rows)
        finals.append(vals[:, 1])
        final_bool.append(sec[cols, 0] == GT)
        j = int(np.argmax(s))
        if s[j] > best_score:
            best_score = float(s[j])
            best_prog = ([tuple(steps[j, 0]), tuple(steps[j, 1])], list(vals[j]))
    problem.counters.programs_scored += total
    if best_prog is None:
        return BaselineOutcome(None, None, time.perf_counter() - start, 0)
    state = space.root()
    for act in best_prog[0]:
        state = space.child(state, as_action(act))
    return BaselineOutcome(
        state,
        best_score,
        time.perf_counter() - start,
        total,
        candidate_values=np.concatenate(finals),
        candidate_bool=np.concatenate(final_bool),
    )
