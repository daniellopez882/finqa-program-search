"""An offline end-to-end run on the 50 FinQA training questions shipped with the package.

It needs no download: retrieval is BM25 only (IDF fitted on the 50 pages), the prior and
the ranker are fitted on 40 questions with small settings, and the other 10 are solved.
The numbers it prints say nothing about FinQA accuracy; it exists to show the whole path
working (CI and the Docker smoke test run it).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from finqa_search.context import ContextConfig
from finqa_search.dataset import Example, from_raw
from finqa_search.evaluate import SearchSetup, run_split, summarise
from finqa_search.pipeline import TrainSetup, train
from finqa_search.retrieval.retriever import FactRetriever, Ranking, RetrieverConfig, fit_idf
from finqa_search.search.mcts import MCTSConfig
from finqa_search.solver import Method, solve, trace_record
from finqa_search.training import PriorConfig, RankerConfig, make_item
from finqa_search.viewer import write_html

N_TRAIN = 40


def load_fixture() -> list[Example]:
    text = resources.files("finqa_search").joinpath("fixtures/finqa50.json").read_text("utf-8")
    return [from_raw(r) for r in json.loads(text)]


def bm25_rankings(examples: list[Example]) -> dict[str, Ranking]:
    retriever = FactRetriever(fit_idf(examples), RetrieverConfig(use_cross_encoder=False))
    return {e.id: retriever.rank(e) for e in examples}


@dataclass(frozen=True)
class DemoSetup:
    train: TrainSetup = field(
        default_factory=lambda: TrainSetup(
            context=ContextConfig(),
            max_steps=4,
            prior=PriorConfig(negatives=24, rounds=40, min_data_in_leaf=10, num_leaves=15),
            ranker=RankerConfig(
                folds=2,
                beam=4,
                samples=4,
                uniform=12,
                rounds=40,
                min_data_in_leaf=10,
                search_simulations=16,
            ),
        )
    )
    methods: tuple[Method, ...] = (
        Method("greedy"),
        Method("beam", width=5),
        Method("mcts", mcts=MCTSConfig(simulations=32)),
    )


def run_demo(
    out_dir: Path, setup: DemoSetup | None = None, log: Callable[[str], None] = print
) -> dict[str, Any]:
    cfg = setup or DemoSetup()
    examples = load_fixture()
    rankings = bm25_rankings(examples)
    train_set, eval_set = examples[:N_TRAIN], examples[N_TRAIN:]
    models = out_dir / "models"
    manifest = train(train_set, rankings, cfg.train, models, workers=1, log=log)
    search = SearchSetup(models / "prior.txt", models / "ranker.txt", max_steps=cfg.train.max_steps)
    results = run_split(eval_set, rankings, cfg.methods, search, workers=1)
    summary = {label: summarise(records) for label, records in results.items()}
    for label, s in summary.items():
        log(
            f"{label:>12}: {s['execution_accuracy'] * len(eval_set):.0f}/{len(eval_set)} "
            f"correct, p50 {s['seconds_p50']:.3f} s"
        )

    prior, ranker = search.load()
    item = make_item(eval_set[0], rankings[eval_set[0].id], cfg.train.context, cfg.train.max_steps)
    mcts = next(m for m in cfg.methods if m.name == "mcts")
    record = trace_record(item, solve(item, mcts, prior, ranker, with_trace=True))
    trace_path = out_dir / "trace.json"
    trace_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    html_path = write_html(record, out_dir / "trace.html")
    log(f"search trace: {trace_path}")
    log(f"tree viewer:  {html_path}")
    return {
        "train": manifest,
        "summary": summary,
        "trace": str(trace_path),
        "html": str(html_path),
    }
