"""PUCT Monte Carlo tree search with progressive widening, transpositions and virtual loss.

Selection at a node with N visits picks, among its first k(N) = ceil(c_pw (N + 1)^alpha)
children in prior order, the one maximising

    Q(child) + c_puct * P(edge) * sqrt(N + 1) / (1 + N(edge))

Nodes live in a transposition table keyed by the problem's canonical state key, so
``add(a, b)`` and ``add(b, a)``, or the same sub-expressions built in another order, share
one node (statistics on nodes, visit counts on edges). Leaves are collected ``batch`` at a
time under virtual loss and expanded and evaluated together, which turns many small model
calls into a few large ones. Q mixes the mean and the max of backed-up values, because the
goal is the single best program, not the average one.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from finqa_search.search.problem import Expansion, SearchProblem


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 64
    c_puct: float = 1.5
    pw_c: float = 2.0
    pw_alpha: float = 0.5
    fpu_reduction: float = 0.0
    batch: int = 8
    max_backup: float = 0.5
    root_noise: float = 0.0
    dirichlet_alpha: float = 0.3
    seed: int = 0
    select: Literal["best_value", "most_visited"] = "best_value"


class Node[S]:
    __slots__ = (
        "children",
        "depth",
        "edge_n",
        "edge_vl",
        "expansion",
        "id",
        "key",
        "n",
        "pending",
        "state",
        "terminal",
        "value",
        "vl",
        "vmax",
        "w",
    )

    def __init__(self, node_id: int, key: str, state: S, depth: int, terminal: bool) -> None:
        self.id = node_id
        self.key = key
        self.state = state
        self.depth = depth
        self.terminal = terminal
        self.expansion: Expansion | None = None
        self.children: list[Node[S] | None] = []
        self.edge_n = np.zeros(0)
        self.edge_vl = np.zeros(0)
        self.n = 0
        self.w = 0.0
        self.vmax = -math.inf
        self.vl = 0
        self.value: float | None = None
        self.pending = False

    def q(self, max_weight: float) -> float | None:
        visits = self.n + self.vl
        if visits == 0:
            return None
        mean = self.w / visits
        if self.n == 0:
            return mean
        return (1.0 - max_weight) * mean + max_weight * self.vmax


@dataclass
class SearchOutcome[S]:
    state: S | None
    score: float | None
    simulations: int
    nodes: int
    seconds: float
    root: Node[S] | None = None
    table: dict[str, Node[S]] = field(default_factory=dict)


class MCTS[S]:
    def __init__(self, problem: SearchProblem[S], config: MCTSConfig) -> None:
        self.problem = problem
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.table: dict[str, Node[S]] = {}
        self._next_id = 0

    def _node(self, key: str, state: S, depth: int, terminal: bool) -> Node[S]:
        node = self.table.get(key)
        if node is None:
            node = Node(self._next_id, key, state, depth, terminal)
            self._next_id += 1
            self.table[key] = node
        return node

    # -- selection -------------------------------------------------------------------------

    def _pick(self, node: Node[S]) -> int:
        exp = node.expansion
        assert exp is not None
        visits = node.n + node.vl
        k = min(len(exp), max(1, math.ceil(self.cfg.pw_c * (visits + 1) ** self.cfg.pw_alpha)))
        parent_q = node.q(self.cfg.max_backup)
        # First-play urgency: an unvisited child is valued at its parent's Q minus a
        # reduction. With no reduction the search tends to try every child once before
        # revisiting any; a large one leaves exploration to the prior term.
        fpu = max(0.0, parent_q - self.cfg.fpu_reduction) if parent_q is not None else 0.0
        q = np.full(k, fpu)
        for i in range(k):
            child = node.children[i]
            if child is not None:
                cq = child.q(self.cfg.max_backup)
                if cq is not None:
                    q[i] = cq
        u = (
            self.cfg.c_puct
            * exp.priors[:k]
            * math.sqrt(visits + 1)
            / (1.0 + node.edge_n[:k] + node.edge_vl[:k])
        )
        return int(np.argmax(q + u))

    def _descend(self) -> tuple[list[tuple[Node[S], int]], Node[S]]:
        node = self.root
        path: list[tuple[Node[S], int]] = []
        while not node.terminal and node.expansion is not None and not node.pending:
            i = self._pick(node)
            path.append((node, i))
            child = node.children[i]
            if child is None:
                action = node.expansion.actions[i]
                if action == node.expansion.stop:
                    child = self._node(node.key + "|stop", node.state, node.depth, True)
                else:
                    state = self.problem.child(node.state, action)
                    child = self._node(self.problem.key(state), state, node.depth + 1, False)
                node.children[i] = child
            node = child
        return path, node

    # -- evaluation ------------------------------------------------------------------------

    def _install(self, node: Node[S], exp: Expansion, value: float) -> None:
        node.pending = False
        if len(exp) == 0:
            node.terminal = True
            node.value = 0.0
            return
        node.expansion = exp
        node.children = [None] * len(exp)
        node.edge_n = np.zeros(len(exp))
        node.edge_vl = np.zeros(len(exp))
        node.value = value

    def _root_noise(self) -> None:
        exp = self.root.expansion
        if exp is None or self.cfg.root_noise <= 0 or len(exp) < 2:
            return
        noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(exp))
        mixed = (1 - self.cfg.root_noise) * exp.priors + self.cfg.root_noise * noise
        order = np.argsort(-mixed, kind="stable")
        self.root.expansion = Expansion([exp.actions[i] for i in order], mixed[order], exp.stop)

    def _evaluate(self, new_nodes: list[Node[S]], stop_nodes: list[Node[S]]) -> None:
        if stop_nodes:
            for node, v in zip(
                stop_nodes,
                self.problem.stop_values([n.state for n in stop_nodes]),
                strict=True,
            ):
                node.value = v
        if new_nodes:
            states = [n.state for n in new_nodes]
            exps = self.problem.expand(states)
            values = self.problem.leaf_values(states, exps)
            for node, exp, v in zip(new_nodes, exps, values, strict=True):
                self._install(node, exp, v)

    @staticmethod
    def _apply_virtual_loss(path: list[tuple[Node[S], int]], leaf: Node[S], sign: int) -> None:
        for node, i in path:
            node.vl += sign
            node.edge_vl[i] += sign
        leaf.vl += sign

    def _backup(self, path: list[tuple[Node[S], int]], leaf: Node[S], value: float) -> None:
        for node, i in path:
            node.n += 1
            node.w += value
            node.vmax = max(node.vmax, value)
            node.edge_n[i] += 1
        leaf.n += 1
        leaf.w += value
        leaf.vmax = max(leaf.vmax, value)

    # -- driver ----------------------------------------------------------------------------

    def run(self) -> SearchOutcome[S]:
        start = time.perf_counter()
        root_state = self.problem.root()
        self.root = self._node(self.problem.key(root_state), root_state, 0, False)
        self.root.pending = True
        self._evaluate([self.root], [])
        self._root_noise()
        if self.root.terminal:
            return SearchOutcome(None, None, 0, len(self.table), time.perf_counter() - start)
        done = 0
        while done < self.cfg.simulations:
            batch: list[tuple[list[tuple[Node[S], int]], Node[S]]] = []
            new_nodes: list[Node[S]] = []
            stop_nodes: list[Node[S]] = []
            for _ in range(min(self.cfg.batch, self.cfg.simulations - done)):
                path, leaf = self._descend()
                self._apply_virtual_loss(path, leaf, +1)
                batch.append((path, leaf))
                done += 1
                if leaf.terminal and leaf.value is None and leaf not in stop_nodes:
                    stop_nodes.append(leaf)
                elif not leaf.terminal and leaf.expansion is None and not leaf.pending:
                    leaf.pending = True
                    new_nodes.append(leaf)
            self._evaluate(new_nodes, stop_nodes)
            for path, leaf in batch:
                self._apply_virtual_loss(path, leaf, -1)
                assert leaf.value is not None
                self._backup(path, leaf, leaf.value)
        state, score = self._final()
        return SearchOutcome(
            state, score, done, len(self.table), time.perf_counter() - start, self.root, self.table
        )

    def _final(self) -> tuple[S | None, float | None]:
        if self.cfg.select == "most_visited":
            node = self.root
            while node.expansion is not None and len(node.expansion) and node.edge_n.sum() > 0:
                i = int(np.argmax(node.edge_n))
                child = node.children[i]
                if child is None:
                    break
                if child.terminal:
                    if child.value is not None and self.problem.can_stop(child.state):
                        return child.state, child.value
                    break
                node = child
            if self.problem.can_stop(node.state):
                return node.state, self.problem.stop_values([node.state])[0]
        found = self.problem.best()
        return (found[0], found[1]) if found else (None, None)


def tree_trace(
    outcome: SearchOutcome[Any],
    describe: Callable[[Any], str],
    describe_action: Callable[[Any, Hashable], str],
    stop_score: Callable[[Any], float | None],
    max_nodes: int = 400,
    unvisited_per_node: int = 3,
) -> dict[str, Any]:
    """Nodes with at least one visit, their edges and priors, for the HTML viewer."""
    if outcome.root is None:
        return {"nodes": [], "root": None}
    nodes: list[dict[str, Any]] = []
    seen: set[int] = set()
    queue = [outcome.root]
    while queue and len(nodes) < max_nodes:
        node = queue.pop(0)
        if node.id in seen:
            continue
        seen.add(node.id)
        edges: list[dict[str, Any]] = []
        if node.expansion is not None:
            unvisited = 0
            for i, action in enumerate(node.expansion.actions):
                child = node.children[i]
                visits = int(node.edge_n[i])
                if visits == 0:
                    if unvisited >= unvisited_per_node:
                        continue
                    unvisited += 1
                edge: dict[str, Any] = {
                    "action": describe_action(node.state, action),
                    "prior": round(float(node.expansion.priors[i]), 6),
                    "visits": visits,
                    "child": child.id if child is not None and visits else None,
                }
                edges.append(edge)
                if child is not None and visits:
                    queue.append(child)
        q = node.q(0.0)
        nodes.append(
            {
                "id": node.id,
                "program": describe(node.state) + (" -> stop" if node.terminal else ""),
                "depth": node.depth,
                "visits": node.n,
                "mean_value": round(q, 6) if q is not None else None,
                "max_value": round(node.vmax, 6) if node.n else None,
                "leaf_value": round(node.value, 6) if node.value is not None else None,
                "stop_score": stop_score(node.state),
                "terminal": node.terminal,
                "edges": edges,
            }
        )
    return {"root": outcome.root.id, "nodes": nodes}
