"""End to end on the bundled 50-question fixture: fit, search, trace, viewer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finqa_search.context import ContextConfig
from finqa_search.dataset import report_of
from finqa_search.demo import N_TRAIN, bm25_rankings, load_fixture
from finqa_search.evaluate import SearchSetup, summarise
from finqa_search.gold import check_split, load_allowlist
from finqa_search.search.mcts import MCTSConfig
from finqa_search.solver import Method, solve
from finqa_search.training import make_item
from finqa_search.viewer import mermaid


def test_fixture_has_fifty_distinct_pages_whose_gold_programs_run() -> None:
    examples = load_fixture()
    assert len(examples) == 50
    assert len({report_of(e.id) for e in examples}) == 50
    report = check_split(examples, load_allowlist())
    assert report.unexpected == {}
    assert report.passed == 50


def test_fixture_train_and_eval_parts_share_no_page() -> None:
    examples = load_fixture()
    train = {report_of(e.id) for e in examples[:N_TRAIN]}
    held = {report_of(e.id) for e in examples[N_TRAIN:]}
    assert not train & held


def test_demo_writes_models_trace_and_viewer(demo_dir: Path) -> None:
    for name in ("models/prior.txt", "models/ranker.txt", "models/manifest.json"):
        assert (demo_dir / name).is_file()
    record = json.loads((demo_dir / "trace.json").read_text(encoding="utf-8"))
    assert record["method"] == "mcts-32"
    assert record["simulations"] == 32
    tree = record["trace"]
    root = next(n for n in tree["nodes"] if n["id"] == tree["root"])
    assert root["visits"] == 32
    assert sum(e["visits"] for e in root["edges"]) == 32
    assert all(0.0 <= e["prior"] <= 1.0 for e in root["edges"])
    assert record["retrieved_facts"]
    assert record["leaves"]
    page = (demo_dir / "trace.html").read_text(encoding="utf-8")
    assert '<script id="trace" type="application/json">' in page
    assert record["question"].split()[0] in page
    sketch = mermaid(record)
    assert sketch.startswith("flowchart LR")
    assert f"n{tree['root']}[" in sketch


@pytest.mark.parametrize(
    "method",
    [
        Method("greedy"),
        Method("beam", width=3),
        Method("beam+ranker", width=3),
        Method("depth2+ranker"),
        Method("mcts", mcts=MCTSConfig(simulations=16)),
        Method("mcts", mcts=MCTSConfig(simulations=16), trees=2),
    ],
    ids=lambda m: m.label,
)
def test_every_method_returns_an_executable_program(demo_dir: Path, method: Method) -> None:
    examples = load_fixture()[N_TRAIN : N_TRAIN + 3]
    rankings = bm25_rankings(load_fixture())
    setup = SearchSetup(demo_dir / "models/prior.txt", demo_dir / "models/ranker.txt")
    prior, ranker = setup.load()
    records = []
    for ex in examples:
        item = make_item(ex, rankings[ex.id], ContextConfig(), 4)
        sol = solve(item, method, prior, ranker)
        assert sol.method == method.label
        if sol.program is not None:
            assert sol.answer is not None
        if method.name == "mcts":
            assert sol.simulations == 16
        records.append(sol.record())
    summary = summarise(records)
    assert summary["n"] == 3
    assert 0.0 <= summary["execution_accuracy"] <= 1.0


def test_mcts_is_deterministic_for_a_seed(demo_dir: Path) -> None:
    ex = load_fixture()[N_TRAIN + 1]
    ranking = bm25_rankings(load_fixture())[ex.id]
    prior, ranker = SearchSetup(
        demo_dir / "models/prior.txt", demo_dir / "models/ranker.txt"
    ).load()
    method = Method("mcts", mcts=MCTSConfig(simulations=24, root_noise=0.25, seed=3))
    runs = [
        solve(make_item(ex, ranking, ContextConfig(), 4), method, prior, ranker, with_trace=True)
        for _ in range(2)
    ]
    assert runs[0].program == runs[1].program
    assert runs[0].trace == runs[1].trace
