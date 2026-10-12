"""Run search methods over a split in worker processes and summarise the results."""

from __future__ import annotations

import multiprocessing as mp
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from finqa_search.context import ContextConfig
from finqa_search.dataset import Example
from finqa_search.dsl import answers_match, equivalent
from finqa_search.prior import LightGBMPrior, Prior, UniformPrior
from finqa_search.ranker import ProgramRanker
from finqa_search.retrieval.retriever import Ranking
from finqa_search.search.parallel import TreeResult, merge, run_tree
from finqa_search.search.problem import ProgramProblem
from finqa_search.solver import Method, any_correct, solve
from finqa_search.training import make_item, parallel_map


@dataclass(frozen=True)
class SearchSetup:
    """Everything a worker needs to rebuild the searcher (picklable, no models inside)."""

    prior_path: Path | None
    ranker_path: Path
    temperature: float = 1.0
    context: ContextConfig = field(default_factory=ContextConfig)
    max_steps: int = 4

    def load(self) -> tuple[Prior, ProgramRanker]:
        prior: Prior
        if self.prior_path is None:
            prior = UniformPrior()
        else:
            prior = LightGBMPrior.load(self.prior_path, self.temperature)
        return prior, ProgramRanker.load(self.ranker_path)


_WORKER: dict[str, Any] = {}


def _init(setup: SearchSetup) -> None:
    prior, ranker = setup.load()
    _WORKER.update(setup=setup, prior=prior, ranker=ranker)


def _job(job: tuple[Example, Ranking, Sequence[Method], bool]) -> list[dict[str, Any]]:
    example, ranking, methods, with_trace = job
    setup: SearchSetup = _WORKER["setup"]
    start = time.perf_counter()
    item = make_item(example, ranking, setup.context, setup.max_steps)
    context_seconds = time.perf_counter() - start
    out = []
    for m in methods:
        rec = solve(item, m, _WORKER["prior"], _WORKER["ranker"], with_trace=with_trace).record()
        rec["seconds_context"] = context_seconds
        out.append(rec)
    return out


def run_split(
    examples: Sequence[Example],
    rankings: dict[str, Ranking],
    methods: Sequence[Method],
    setup: SearchSetup,
    workers: int = 1,
    with_trace: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Every method on every question; one question (all its methods) per task."""
    labels = [m.label for m in methods]
    if len(set(labels)) != len(labels):
        raise ValueError(f"method labels must be unique, got {labels}")
    jobs = [(e, rankings[e.id], list(methods), with_trace) for e in examples]
    per_question = parallel_map(_job, jobs, workers, _init, (setup,))
    out: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    for records in per_question:
        for rec in records:
            out[rec["method"]].append(rec)
    return out


def percentile(xs: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q)) if xs else float("nan")


def summarise(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if not n:
        return {"n": 0}
    secs = [r["seconds"] for r in records]
    reach = [r for r in records if r["gold_reachable"]]

    def rate(rs: Sequence[dict[str, Any]], key: str) -> float:
        return sum(1 for r in rs if r[key]) / len(rs) if rs else float("nan")

    out: dict[str, Any] = {
        "n": n,
        "execution_accuracy": rate(records, "exe_correct"),
        "program_accuracy": rate(records, "prog_correct"),
        "candidate_coverage": rate(records, "candidates_correct"),
        "gold_reachable": len(reach) / n,
        "execution_accuracy_when_reachable": rate(reach, "exe_correct"),
        "no_answer": sum(1 for r in records if r["program"] is None) / n,
        "seconds_mean": statistics.fmean(secs),
        "seconds_p50": percentile(secs, 50),
        "seconds_p95": percentile(secs, 95),
    }
    for key in ("expansions", "prior_rows", "programs_scored", "simulations", "nodes"):
        if all(key in r for r in records):
            out[f"{key}_mean"] = statistics.fmean(r[key] for r in records)
    if all("seconds_context" in r for r in records):
        out["seconds_context_p50"] = percentile([r["seconds_context"] for r in records], 50)
    return out


# -- latency with root parallelism ----------------------------------------------------


def _tree_job(job: tuple[Example, Ranking, Method, int]) -> TreeResult:
    example, ranking, method, index = job
    setup: SearchSetup = _WORKER["setup"]
    item = make_item(example, ranking, setup.context, setup.max_steps)
    problem = ProgramProblem(
        item.space,
        _WORKER["prior"],
        _WORKER["ranker"],
        lookahead=method.lookahead,
        prior_weight=method.prior_weight,
    )
    return run_tree(problem, method.mcts, index, method.trees)


def _ready(_: int) -> bool:
    return "ranker" in _WORKER


def run_latency(
    examples: Sequence[Example],
    rankings: dict[str, Ranking],
    method: Method,
    setup: SearchSetup,
    workers: int,
) -> list[dict[str, Any]]:
    """Answer questions one at a time, the trees of each spread over ``workers`` processes.

    The clock for a question runs from handing its trees to the pool until the merged
    answer is back, so it includes building the question context in every worker and
    moving the results between processes. Grading happens after the clock stops.
    """
    ctx = mp.get_context("spawn")
    records: list[dict[str, Any]] = []
    with ctx.Pool(workers, initializer=_init, initargs=(setup,)) as pool:
        if not all(pool.map(_ready, range(workers * 4), chunksize=1)):
            raise RuntimeError("a worker failed to load the models")
        for ex in examples:
            jobs = [(ex, rankings[ex.id], method, i) for i in range(method.trees)]
            start = time.perf_counter()
            results = pool.map(_tree_job, jobs, chunksize=1)
            merged = merge(results, time.perf_counter() - start)
            records.append(_grade_merged(ex, rankings[ex.id], method, setup, merged, workers))
    return records


def _grade_merged(
    ex: Example,
    ranking: Ranking,
    method: Method,
    setup: SearchSetup,
    merged: Any,
    workers: int,
) -> dict[str, Any]:
    item = make_item(ex, ranking, setup.context, setup.max_steps)
    program = answer = None
    exe_ok = prog_ok = False
    if merged.state is not None:
        prog = item.space.program(merged.state)
        program, answer = prog.render(), merged.state.answer
        exe_ok = answers_match(answer, ex.exe_ans)
        gold = ex.gold_program()
        prog_ok = gold is not None and equivalent(prog, gold)
    return {
        "id": ex.id,
        "method": method.label,
        "workers": workers,
        "trees": method.trees,
        "program": program,
        "answer": answer,
        "exe_correct": exe_ok,
        "prog_correct": prog_ok,
        "candidates_correct": any_correct((s.answer for s in merged.finished()), ex.exe_ans),
        "gold_reachable": item.gold is not None,
        "seconds": merged.seconds,
        "simulations": merged.simulations,
        "nodes": merged.nodes,
        "tree_seconds": [t.seconds for t in merged.trees],
    }


def progress_printer(label: str, every: int = 100) -> Callable[[int, int], None]:
    def show(i: int, total: int) -> None:
        if i % every == 0 or i == total:
            print(f"  {label}: {i}/{total}", flush=True)

    return show
