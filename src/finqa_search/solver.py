"""Run one search method on one question and grade the answer."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import numpy as np

from finqa_search.dsl import Answer, answers_match, equivalent, gold_value
from finqa_search.prior import Prior
from finqa_search.ranker import ProgramRanker
from finqa_search.search.baselines import beam, exhaustive_depth2, greedy
from finqa_search.search.mcts import MCTS, MCTSConfig, tree_trace
from finqa_search.search.parallel import run_root_parallel
from finqa_search.search.problem import Counters, ProgramProblem, as_action, log_prob
from finqa_search.space import STOP_ACTION, State
from finqa_search.training import Item

MethodName = Literal["greedy", "beam", "beam+ranker", "depth2+ranker", "mcts"]


@dataclass(frozen=True)
class Method:
    name: MethodName
    width: int = 0
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    lookahead: int = 3
    trees: int = 1  # root parallelism: the simulation budget is split over this many trees
    prior_weight: float = 1.0  # weight of log P_prior when picking the answer
    tag: str = ""  # overrides the label, to tell apart settings of one method

    @property
    def label(self) -> str:
        if self.tag:
            return self.tag
        if self.name in ("beam", "beam+ranker"):
            return f"{self.name}-{self.width}"
        if self.name == "mcts":
            trees = f"x{self.trees}" if self.trees > 1 else ""
            return f"mcts-{self.mcts.simulations}{trees}"
        return self.name

    def with_seed(self, seed: int) -> Method:
        return replace(self, mcts=replace(self.mcts, seed=seed))


@dataclass
class Solution:
    id: str
    method: str
    program: str | None
    answer: float | bool | None
    exe_correct: bool
    prog_correct: bool
    gold_reachable: bool
    candidates_correct: bool  # some program the method scored or finished was right
    score: float | None  # what the answer was picked by: log P_ranker (+ weighted log prior)
    seconds: float
    expansions: int
    prior_rows: int
    programs_scored: int
    seconds_prior: float
    seconds_ranker: float
    simulations: int = 0
    nodes: int = 0
    trace: dict[str, Any] | None = None

    def record(self) -> dict[str, Any]:
        out = asdict(self)
        if out["trace"] is None:
            out.pop("trace")
        return out


def any_correct(answers: Iterable[Answer | None], gold: float | str) -> bool:
    return any(answers_match(a, gold) for a in answers)


def values_contain(
    values: np.ndarray | None, is_bool: np.ndarray | None, gold: float | str
) -> bool:
    """Whether a batch of final values (yes/no ones marked) holds the gold answer."""
    g = gold_value(gold)
    if values is None or is_bool is None or g is None:
        return False
    if isinstance(g, bool):
        return bool(np.any(is_bool & (values == float(g))))
    near = values[~is_bool & (np.abs(values - g) <= 1e-5)]
    return any(answers_match(float(v), gold) for v in near)


def describe(item: Item, state: State) -> str:
    return item.space.program(state).render() if state.steps else "(empty)"


def describe_action(item: Item, state: State, action: object) -> str:
    act = as_action(action)  # type: ignore[arg-type]
    if act == STOP_ACTION:
        return "stop"
    return item.space.step(state, act).render()


def solve(
    item: Item,
    method: Method,
    prior: Prior,
    ranker: ProgramRanker,
    with_trace: bool = False,
) -> Solution:
    def new_problem() -> ProgramProblem:
        return ProgramProblem(
            item.space, prior, ranker, lookahead=method.lookahead, prior_weight=method.prior_weight
        )

    problem = new_problem()
    counters = problem.counters
    start = time.perf_counter()
    trace: dict[str, Any] | None = None
    sims = nodes = 0
    gold = item.example.exe_ans
    if method.name == "greedy":
        out = greedy(problem)
        state, score = out.state, out.score
        found = any_correct((s.answer for s in out.candidates), gold)
    elif method.name in ("beam", "beam+ranker"):
        out = beam(problem, method.width, rerank=method.name == "beam+ranker")
        state, score = out.state, out.score
        found = any_correct((s.answer for s in out.candidates), gold)
    elif method.name == "depth2+ranker":
        out = exhaustive_depth2(problem)
        state = out.state
        score = None if out.score is None else log_prob(out.score)
        found = values_contain(out.candidate_values, out.candidate_bool, gold)
    elif method.trees > 1:
        merged = run_root_parallel(new_problem, method.mcts, method.trees)
        state, score = merged.state, merged.score
        sims, nodes, counters = merged.simulations, merged.nodes, merged.counters
        found = any_correct((s.answer for s in merged.finished()), gold)
    else:
        search = MCTS(problem, method.mcts)
        res = search.run()
        state, score, sims, nodes = res.state, res.score, res.simulations, res.nodes
        found = any_correct((s.answer for s, _ in problem.finished().values()), gold)
        if with_trace:
            trace = tree_trace(
                res,
                lambda s: describe(item, s),
                lambda s, a: describe_action(item, s, a),
                lambda s: round(problem.scored[s.key][1], 6) if s.key in problem.scored else None,
            )
    seconds = time.perf_counter() - start

    program_text = answer = None
    exe_ok = prog_ok = False
    if state is not None and state.steps:
        program = item.space.program(state)
        program_text = program.render()
        answer = state.answer
        exe_ok = answers_match(answer, item.example.exe_ans)
        gold = item.example.gold_program()
        prog_ok = gold is not None and equivalent(program, gold)
    c: Counters = counters
    return Solution(
        id=item.example.id,
        method=method.label,
        program=program_text,
        answer=answer,
        exe_correct=exe_ok,
        prog_correct=prog_ok,
        gold_reachable=item.gold is not None,
        candidates_correct=found,
        score=score,
        seconds=seconds,
        expansions=c.expansions,
        prior_rows=c.prior_rows,
        programs_scored=c.programs_scored,
        seconds_prior=c.seconds_prior,
        seconds_ranker=c.seconds_ranker,
        simulations=sims,
        nodes=nodes,
        trace=trace,
    )


def trace_record(item: Item, solution: Solution, top_facts: int = 5) -> dict[str, Any]:
    """A solution plus what a reader needs to judge it: question, gold, evidence, leaves."""
    ex = item.example
    facts = {f.id: f.text for f in ex.facts()}
    ranked = sorted(item.ctx.fact_rank, key=item.ctx.fact_rank.__getitem__)[:top_facts]
    out = solution.record()
    out.update(
        question=ex.question,
        gold_program=ex.program,
        gold_answer=ex.exe_ans,
        gold_facts=sorted(ex.gold_inds),
        retrieved_facts=[{"id": f, "text": facts.get(f, "")} for f in ranked],
        leaves=[
            {
                "arg": leaf.arg.render(),
                "value": leaf.value,
                "source": leaf.mention.fact_id if leaf.mention is not None else "constant",
            }
            for leaf in item.ctx.leaves
        ],
        rows=[row.label for row in item.ctx.rows],
    )
    return out
