"""PUCT search on toy problems whose optimum is known."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field

import numpy as np
import pytest

from finqa_search.search.mcts import MCTS, MCTSConfig, Node
from finqa_search.search.parallel import merge, run_root_parallel, run_tree, split_budget
from finqa_search.search.problem import Counters, Expansion

Path = tuple[int, ...]
STOP = "stop"


@dataclass
class ToyProblem:
    """Pick ``depth`` digits in ``range(branching)``; finishing scores ``reward(path)``.

    With ``commutative`` set, the key ignores the order of the digits, so permutations of
    one path are transpositions of one node.
    """

    depth: int
    branching: int
    reward: Callable[[Path], float]
    prior: Callable[[Path], np.ndarray] | None = None
    commutative: bool = False
    counters: Counters = field(default_factory=Counters)
    scored: dict[str, tuple[Path, float]] = field(default_factory=dict)

    def root(self) -> Path:
        return ()

    def key(self, state: Path) -> str:
        return repr(tuple(sorted(state)) if self.commutative else state)

    def can_stop(self, state: Path) -> bool:
        return len(state) == self.depth

    def expand(self, states: list[Path]) -> list[Expansion]:
        out = []
        for s in states:
            self.counters.expansions += 1
            if len(s) == self.depth:
                out.append(Expansion([STOP], np.ones(1), STOP))
                continue
            p = self.prior(s) if self.prior else np.full(self.branching, 1.0 / self.branching)
            order = np.argsort(-p, kind="stable")
            out.append(Expansion([int(i) for i in order], p[order], STOP))
        return out

    def child(self, state: Path, action: Hashable) -> Path:
        assert isinstance(action, int)
        return (*state, action)

    def stop_values(self, states: list[Path]) -> list[float]:
        out = []
        for s in states:
            v = self.reward(s)
            self.scored[self.key(s)] = (s, v)
            out.append(v)
        return out

    def leaf_values(self, states: list[Path], expansions: list[Expansion]) -> list[float]:
        return [0.0 for _ in states]

    def finished(self) -> dict[str, tuple[Path, float]]:
        return dict(self.scored)

    def best(self) -> tuple[Path, float] | None:
        return max(self.scored.values(), key=lambda kv: kv[1]) if self.scored else None


TARGET = (2, 0, 3)


def closeness(path: Path) -> float:
    """1.0 at TARGET, partial credit per matching digit elsewhere."""
    hits = sum(a == b for a, b in zip(path, TARGET, strict=True))
    return 1.0 if path == TARGET else 0.25 * hits


def test_finds_the_optimum_with_a_uniform_prior() -> None:
    problem = ToyProblem(depth=3, branching=4, reward=closeness)
    out = MCTS(problem, MCTSConfig(simulations=400, batch=4)).run()
    assert out.state == TARGET
    assert out.score == 1.0
    assert out.simulations == 400


def test_value_backups_overrule_a_misleading_prior() -> None:
    def bad_prior(path: Path) -> np.ndarray:
        p = np.full(4, 0.2)
        p[1] = 0.4  # digit 1 is never right
        return p

    problem = ToyProblem(depth=3, branching=4, reward=closeness, prior=bad_prior)
    search = MCTS(problem, MCTSConfig(simulations=400, batch=4))
    out = search.run()
    assert out.state == TARGET
    root = search.root
    assert root.expansion is not None
    most_visited = root.expansion.actions[int(np.argmax(root.edge_n))]
    assert most_visited == TARGET[0]


def test_a_good_prior_needs_fewer_simulations() -> None:
    def good_prior(path: Path) -> np.ndarray:
        p = np.full(5, 0.02)
        p[TARGET[len(path)]] = 0.92
        return p

    guided = ToyProblem(depth=3, branching=5, reward=closeness, prior=good_prior)
    blind = ToyProblem(depth=3, branching=5, reward=closeness)
    assert MCTS(guided, MCTSConfig(simulations=12, batch=1)).run().state == TARGET
    assert MCTS(blind, MCTSConfig(simulations=12, batch=1)).run().state != TARGET


def visit_profile(seed: int, noise: float) -> list[tuple[str, int]]:
    problem = ToyProblem(depth=3, branching=4, reward=closeness)
    out = MCTS(problem, MCTSConfig(simulations=120, batch=4, seed=seed, root_noise=noise)).run()
    return sorted((key, node.n) for key, node in out.table.items())


def test_search_is_deterministic_for_a_seed() -> None:
    assert visit_profile(7, 0.3) == visit_profile(7, 0.3)
    assert visit_profile(7, 0.3) != visit_profile(8, 0.3)


def test_transpositions_share_one_node() -> None:
    ordered = ToyProblem(depth=3, branching=4, reward=lambda p: float(sum(p) == 6))
    commut = ToyProblem(depth=3, branching=4, reward=lambda p: float(sum(p) == 6), commutative=True)
    cfg = MCTSConfig(simulations=300, batch=4, pw_c=8.0)
    a = MCTS(ordered, cfg).run()
    b = MCTS(commut, cfg).run()
    assert b.nodes < a.nodes
    assert a.score == b.score == 1.0
    # Two orders of the same digits land on the same node object.
    search = MCTS(commut, cfg)
    search.run()
    n1 = search.table[commut.key((1, 2))]
    assert search.table.get(commut.key((2, 1))) is n1


def make_node(priors: list[float], edge_n: list[int], child_q: list[float | None]) -> Node[Path]:
    node: Node[Path] = Node(0, "root", (), 0, False)
    node.expansion = Expansion(list(range(len(priors))), np.array(priors), STOP)
    node.edge_n = np.array(edge_n, dtype=float)
    node.edge_vl = np.zeros(len(priors))
    node.n = sum(edge_n)
    node.w = 0.5 * node.n
    node.vmax = 1.0
    children: list[Node[Path] | None] = []
    for i, q in enumerate(child_q):
        if q is None:
            children.append(None)
            continue
        c: Node[Path] = Node(i + 1, f"c{i}", (i,), 1, False)
        c.n = max(edge_n[i], 1)
        c.w = q * c.n
        c.vmax = q
        children.append(c)
    node.children = children
    return node


def test_pick_maximises_q_plus_puct_exploration() -> None:
    cfg = MCTSConfig(c_puct=1.5, pw_c=100.0, max_backup=0.0)
    search = MCTS(ToyProblem(1, 3, closeness), cfg)
    node = make_node([0.6, 0.3, 0.1], [10, 2, 0], [0.4, 0.5, None])
    picked = search._pick(node)
    fpu = 0.5 * (1 - cfg.max_backup) + cfg.max_backup * 1.0  # parent mean, no max mixing
    sqrt_n = math.sqrt(12 + 1)
    scores = [
        0.4 + 1.5 * 0.6 * sqrt_n / 11,
        0.5 + 1.5 * 0.3 * sqrt_n / 3,
        fpu + 1.5 * 0.1 * sqrt_n / 1,
    ]
    assert picked == int(np.argmax(scores))


def test_q_mixes_mean_and_max() -> None:
    node: Node[Path] = Node(0, "k", (), 0, False)
    assert node.q(0.5) is None
    node.n, node.w, node.vmax = 4, 1.0, 0.9
    assert node.q(0.0) == pytest.approx(0.25)
    assert node.q(1.0) == pytest.approx(0.9)
    assert node.q(0.5) == pytest.approx(0.575)


def test_progressive_widening_limits_the_candidates() -> None:
    cfg = MCTSConfig(c_puct=10.0, pw_c=1.0, pw_alpha=0.5)
    search = MCTS(ToyProblem(1, 6, closeness), cfg)
    # 0 visits: k = ceil(1 * 1^0.5) = 1, so only the first child can be picked even though
    # the exploration bonus is larger for the low-prior unvisited ones.
    node = make_node([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], [0] * 6, [None] * 6)
    assert search._pick(node) == 0
    # 8 visits on child 0: k = ceil(sqrt(9)) = 3, child 0 is now expensive to revisit.
    node = make_node([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], [8, 0, 0, 0, 0, 0], [0.0, *[None] * 5])
    assert search._pick(node) in (1, 2)


def test_virtual_loss_spreads_one_batch_over_several_children() -> None:
    problem = ToyProblem(depth=2, branching=6, reward=lambda p: 0.0)
    search = MCTS(problem, MCTSConfig(simulations=6, batch=6, pw_c=6.0))
    out = search.run()
    root = out.root
    assert root is not None
    assert int((root.edge_n > 0).sum()) > 1
    assert root.vl == 0
    assert root.edge_vl.sum() == 0


def test_root_parallel_trees_split_the_budget_and_merge() -> None:
    assert split_budget(256, 6) == [43, 43, 43, 43, 42, 42]
    assert sum(split_budget(10, 3)) == 10
    with pytest.raises(ValueError, match="at least one tree"):
        split_budget(10, 0)
    merged = run_root_parallel(
        lambda: ToyProblem(depth=3, branching=4, reward=closeness), MCTSConfig(simulations=600), 3
    )
    assert merged.state == TARGET
    assert merged.simulations == 600
    assert len(merged.trees) == 3
    assert sum(merged.root_visits.values()) == sum(
        sum(t.root_visits.values()) for t in merged.trees
    )
    assert merged.counters.expansions == sum(t.counters.expansions for t in merged.trees)


def test_root_parallel_merge_takes_the_best_tree() -> None:
    # Tree 0 runs without root noise and gets half of the budget it is given.
    strong = run_tree(ToyProblem(3, 4, closeness), MCTSConfig(simulations=800), 0, 2)
    weak = run_tree(ToyProblem(3, 4, closeness), MCTSConfig(simulations=16), 1, 2)
    assert (strong.simulations, weak.simulations) == (400, 8)
    merged = merge([weak, strong], seconds=0.0)
    every = [v for t in (weak, strong) for _, v in t.scored.values()]
    assert merged.score == max(every) == 1.0
    assert merged.state == TARGET
