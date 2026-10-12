"""What a search needs from the program space, as a small protocol.

``MCTS`` and the baselines only talk to a ``SearchProblem``. ``ProgramProblem`` wires the
FinQA action space, a prior and the ranker into one; the tests use toy problems with
known optima through the same interface.
"""

from __future__ import annotations

import math
import time
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import numpy as np

from finqa_search.prior import Prior
from finqa_search.ranker import ProgramRanker, pack, program_features
from finqa_search.space import STOP_ACTION, Action, ActionBatch, ActionSpace, State

S = TypeVar("S")


@dataclass
class Expansion:
    """Legal actions of a state with their prior probabilities, best first."""

    actions: Sequence[Hashable]
    priors: np.ndarray
    stop: Hashable | None = None
    _position: dict[Hashable, int] | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.actions)

    def prior_of(self, action: Hashable) -> float:
        if self._position is None:
            self._position = {a: i for i, a in enumerate(self.actions)}
        i = self._position.get(action)
        return 0.0 if i is None else float(self.priors[i])


@dataclass
class Counters:
    expansions: int = 0
    prior_rows: int = 0
    programs_scored: int = 0
    seconds_prior: float = 0.0
    seconds_ranker: float = 0.0


class SearchProblem(Protocol[S]):
    counters: Counters

    def root(self) -> S: ...

    def key(self, state: S) -> str: ...

    def can_stop(self, state: S) -> bool: ...

    def expand(self, states: list[S]) -> list[Expansion]: ...

    def child(self, state: S, action: Hashable) -> S: ...

    def stop_values(self, states: list[S]) -> list[float]:
        """Value of answering with ``state`` as it is (each state must be able to stop)."""
        ...

    def leaf_values(self, states: list[S], expansions: list[Expansion]) -> list[float]:
        """Estimate for a newly expanded state, used as its first backup."""
        ...

    def finished(self) -> dict[str, tuple[S, float]]:
        """Every finished state scored so far, by key, with the score that picks the answer."""
        ...

    def best(self) -> tuple[S, float] | None:
        """The finished state with the highest answer-picking score so far."""
        ...


LOG_PRIOR_FLOOR = -50.0


@dataclass
class ProgramProblem:
    """FinQA programs: prior from ``prior``, value from ``ranker`` with a short lookahead.

    A new node's value is the best ranker score among stopping here and stopping after
    one of its ``lookahead`` most probable next steps. Without the lookahead every
    half-built program ("subtract the two years" before "divide by the base") would
    score as a wrong answer and PUCT would starve the branch that finishes it.

    The search is steered by the ranker alone, but the answer is the finished program
    with the highest ``log P_ranker(correct) + prior_weight * log P_prior(program)``,
    where the prior log-likelihood is that of the most probable derivation seen. A
    search that scores hundreds of programs finds some the ranker over-rates; the prior
    term discounts programs no annotator would plausibly write.
    """

    space: ActionSpace
    prior: Prior
    ranker: ProgramRanker | None
    lookahead: int = 3
    prior_weight: float = 1.0
    counters: Counters = field(default_factory=Counters)
    scored: dict[str, tuple[State, float]] = field(default_factory=dict)
    expanded: dict[str, Expansion] = field(default_factory=dict)
    log_prior: dict[str, float] = field(default_factory=lambda: {State().key: 0.0})

    def root(self) -> State:
        return self.space.root()

    def key(self, state: State) -> str:
        return state.key

    def can_stop(self, state: State) -> bool:
        return state.can_stop

    def child(self, state: State, action: Hashable) -> State:
        act = as_action(action)
        nxt = self.space.child(state, act)
        exp = self.expanded.get(state.key)
        base = self.log_prior.get(state.key)
        if exp is not None and base is not None:
            lp = base + log_prob(exp.prior_of(act))
            if lp > self.log_prior.get(nxt.key, -math.inf):
                self.log_prior[nxt.key] = lp
        return nxt

    def expand(self, states: list[State]) -> list[Expansion]:
        """Prior over the legal actions of each state; memoised by state key, because beam
        search and the ranker's sampler revisit the same prefixes."""
        todo = list({s.key: s for s in states if s.key not in self.expanded}.values())
        if todo:
            start = time.perf_counter()
            batches: list[ActionBatch] = [self.space.legal(s) for s in todo]
            probs = self.prior(self.space, todo, batches)
            for state, batch, p in zip(todo, batches, probs, strict=True):
                order = np.argsort(-p, kind="stable")
                acts: list[Action] = [(op, a, b) for op, a, b in batch.actions[order].tolist()]
                self.expanded[state.key] = Expansion(acts, p[order], STOP_ACTION)
                self.counters.prior_rows += len(batch)
            self.counters.expansions += len(todo)
            self.counters.seconds_prior += time.perf_counter() - start
        return [self.expanded[s.key] for s in states]

    def score_states(self, states: list[State]) -> list[float]:
        """Ranker P(correct) for finishing at each state; memoised by state key."""
        todo = [s for s in states if s.key not in self.scored]
        if todo:
            if self.ranker is None:
                raise RuntimeError("this problem was built without a ranker")
            start = time.perf_counter()
            uniq = list({s.key: s for s in todo}.values())
            progs = [(list(s.steps), list(s.values)) for s in uniq]
            steps, values, lengths = pack(progs, self.space.max_steps)
            X = program_features(self.space.ctx, steps, values, lengths)
            scores = self.ranker.score(X)
            for s, v in zip(uniq, scores, strict=True):
                self.scored[s.key] = (s, float(v))
            self.counters.programs_scored += len(uniq)
            self.counters.seconds_ranker += time.perf_counter() - start
        return [self.scored[s.key][1] for s in states]

    def stop_values(self, states: list[State]) -> list[float]:
        return self.score_states(states)

    def leaf_values(self, states: list[State], expansions: list[Expansion]) -> list[float]:
        groups: list[list[State]] = []
        for state, exp in zip(states, expansions, strict=True):
            group = [state] if state.can_stop else []
            taken = 0
            for act in exp.actions:
                if taken >= self.lookahead:
                    break
                if act == STOP_ACTION:
                    continue
                taken += 1
                nxt = self.child(state, act)
                if nxt.can_stop:
                    group.append(nxt)
            groups.append(group)
        flat = [s for g in groups for s in g]
        scores = iter(self.score_states(flat))
        return [max((next(scores) for _ in g), default=0.0) for g in groups]

    def selection_score(self, state: State) -> float:
        """``log P_ranker + prior_weight * log P_prior`` of a scored state."""
        r = self.scored[state.key][1]
        lp = self.log_prior.get(state.key, LOG_PRIOR_FLOOR)
        return log_prob(r) + self.prior_weight * max(lp, LOG_PRIOR_FLOOR)

    def finished(self) -> dict[str, tuple[State, float]]:
        return {k: (s, self.selection_score(s)) for k, (s, _) in self.scored.items()}

    def best(self) -> tuple[State, float] | None:
        if not self.scored:
            return None
        return max(self.finished().values(), key=lambda kv: kv[1])


def log_prob(p: float) -> float:
    return math.log(max(p, 1e-12))


def as_action(a: Hashable) -> Action:
    op, x, y = a  # type: ignore[misc]
    return (int(op), int(x), int(y))
