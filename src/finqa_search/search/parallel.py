"""Root parallelism: independent PUCT trees over one question, merged at the root.

Each tree gets its own seed and a share of the simulation budget. Trees after the first
mix Dirichlet noise into the root prior so they do not all walk the same path. Trees
share nothing while they run (no locks, no shared table), which is what makes the scheme
work across processes; the price is that transpositions are found once per tree.

The merge keeps every program any tree scored and answers with the best-scoring one,
the same rule a single tree uses, and sums root visit counts per action for reporting.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from finqa_search.search.mcts import MCTS, MCTSConfig
from finqa_search.search.problem import Counters, SearchProblem

NOISE_FOR_EXTRA_TREES = 0.25


def split_budget(total: int, trees: int) -> list[int]:
    """``total`` simulations over ``trees`` trees, as evenly as possible (first trees get more)."""
    if trees < 1:
        raise ValueError("need at least one tree")
    base, extra = divmod(total, trees)
    return [base + (1 if i < extra else 0) for i in range(trees)]


def tree_config(config: MCTSConfig, index: int, trees: int) -> MCTSConfig:
    noise = config.root_noise if index == 0 else max(config.root_noise, NOISE_FOR_EXTRA_TREES)
    return replace(
        config,
        simulations=split_budget(config.simulations, trees)[index],
        seed=config.seed + index,
        root_noise=noise,
    )


@dataclass
class TreeResult:
    """What one tree sends back: scored programs by key, root visits, cost counters."""

    index: int
    simulations: int
    nodes: int
    seconds: float
    scored: dict[str, tuple[Any, float]]
    root_visits: dict[Hashable, int]
    counters: Counters


@dataclass
class MergedOutcome:
    state: Any
    score: float | None
    simulations: int
    nodes: int
    seconds: float
    trees: list[TreeResult] = field(default_factory=list)
    root_visits: dict[Hashable, int] = field(default_factory=dict)

    def finished(self) -> list[Any]:
        """Every program any tree scored."""
        return [state for t in self.trees for state, _ in t.scored.values()]

    @property
    def counters(self) -> Counters:
        total = Counters()
        for t in self.trees:
            for name in ("expansions", "prior_rows", "programs_scored"):
                setattr(total, name, getattr(total, name) + getattr(t.counters, name))
            total.seconds_prior += t.counters.seconds_prior
            total.seconds_ranker += t.counters.seconds_ranker
        return total


def run_tree(
    problem: SearchProblem[Any],
    config: MCTSConfig,
    index: int,
    trees: int,
    portable: Callable[[Any], Any] = lambda s: s,
) -> TreeResult:
    """Run tree ``index`` of ``trees`` on ``problem``.

    ``portable`` turns a state into something cheap to send between processes.
    """
    start = time.perf_counter()
    out = MCTS(problem, tree_config(config, index, trees)).run()
    visits: dict[Hashable, int] = {}
    root = out.root
    if root is not None and root.expansion is not None:
        for i, action in enumerate(root.expansion.actions):
            if root.edge_n[i]:
                visits[action] = int(root.edge_n[i])
    scored = {key: (portable(s), v) for key, (s, v) in problem.finished().items()}
    return TreeResult(
        index=index,
        simulations=out.simulations,
        nodes=out.nodes,
        seconds=time.perf_counter() - start,
        scored=scored,
        root_visits=visits,
        counters=problem.counters,
    )


def merge(results: Sequence[TreeResult], seconds: float) -> MergedOutcome:
    pool: dict[str, tuple[Any, float]] = {}
    visits: dict[Hashable, int] = {}
    for r in results:
        pool.update(r.scored)
        for action, n in r.root_visits.items():
            visits[action] = visits.get(action, 0) + n
    best = max(pool.values(), key=lambda sv: sv[1]) if pool else (None, None)
    return MergedOutcome(
        state=best[0],
        score=best[1],
        simulations=sum(r.simulations for r in results),
        nodes=sum(r.nodes for r in results),
        seconds=seconds,
        trees=list(results),
        root_visits=visits,
    )


def run_root_parallel(
    make_problem: Callable[[], SearchProblem[Any]], config: MCTSConfig, trees: int
) -> MergedOutcome:
    """All trees in this process, one after another (same result as the process pool)."""
    start = time.perf_counter()
    results = [run_tree(make_problem(), config, i, trees) for i in range(trees)]
    return merge(results, time.perf_counter() - start)
