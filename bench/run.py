"""The benchmark behind every number in the README.

    uv run python bench/run.py              # everything, on the downloaded data
    uv run python bench/run.py --smoke      # tiny subsets, a few minutes, smoke-*.json

Each stage writes ``bench/results/<stage>.json`` with the machine, date, git sha and
command next to its metrics; ``--stages`` runs a subset (later stages read what earlier
ones left in the data directory: rankings, models, the tuned settings).

    data       sha256 check, split sizes, every gold program re-executed, split leakage
    retrieval  fact rankings for all splits (BM25 + cross-encoder), Recall@k vs gold
    train      action prior and program ranker, fitted on train
    tune       dev-only grid for MCTS and the answer-picking prior weight
    test       test split: greedy, beam, beam+ranker, depth-2+ranker, MCTS 16..1024
    latency    MCTS-256 per-question seconds with 1 worker and 6 root-parallel workers
    trace      one annotated search tree for the README
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from finqa_search.context import ContextConfig
from finqa_search.dataset import SPLITS, Example, leakage, load_split
from finqa_search.evaluate import (
    SearchSetup,
    progress_printer,
    run_latency,
    run_split,
    summarise,
)
from finqa_search.fetch import verify_all
from finqa_search.gold import check_split, load_allowlist
from finqa_search.layout import idf_path, models_dir, rankings_path, runs_dir
from finqa_search.pipeline import TrainSetup, train
from finqa_search.retrieval.batch import rank_examples
from finqa_search.retrieval.retriever import (
    Ranking,
    RetrieverConfig,
    bm25_only,
    fit_idf,
    load_rankings,
    recall_at_k,
    save_idf,
    save_rankings,
)
from finqa_search.search.mcts import MCTSConfig
from finqa_search.search.parallel import split_budget
from finqa_search.solver import Method, solve, trace_record
from finqa_search.sources import data_dir
from finqa_search.training import PriorConfig, RankerConfig, make_item
from finqa_search.viewer import mermaid
from machine import stamp

RESULTS = Path(__file__).resolve().parent / "results"
STAGES = ("data", "retrieval", "train", "tune", "test", "latency", "trace")


@dataclass(frozen=True)
class Plan:
    """Sizes and settings of one benchmark run (the full one or the smoke one)."""

    prefix: str
    limits: dict[str, int | None]
    train: TrainSetup
    tune_simulations: int
    tune_grid: tuple[tuple[float, float], ...]  # (c_puct, fpu_reduction)
    prior_weights: tuple[float, ...]
    simulations: tuple[int, ...]
    beam_widths: tuple[int, ...]
    latency_questions: int
    latency_simulations: int
    latency_workers: int


FULL = Plan(
    prefix="",
    limits={"train": None, "dev": None, "test": None},
    train=TrainSetup(ContextConfig(), 4, PriorConfig(), RankerConfig()),
    tune_simulations=64,
    tune_grid=((0.5, 0.0), (1.5, 0.0), (4.0, 0.0), (0.5, 1.0), (1.5, 1.0), (4.0, 1.0)),
    prior_weights=(0.0, 0.5, 1.0, 2.0),
    simulations=(16, 64, 256, 1024),
    beam_widths=(5, 10),
    latency_questions=200,
    latency_simulations=256,
    latency_workers=6,
)

SMOKE = Plan(
    prefix="smoke-",
    limits={"train": 300, "dev": 24, "test": 24},
    train=TrainSetup(
        ContextConfig(),
        4,
        PriorConfig(rounds=30, negatives=24),
        RankerConfig(beam=4, samples=4, uniform=12, rounds=30, search_simulations=16),
    ),
    tune_simulations=16,
    tune_grid=((1.5, 0.0), (1.5, 1.0)),
    prior_weights=(1.0,),
    simulations=(16, 64),
    beam_widths=(5,),
    latency_questions=6,
    latency_simulations=32,
    latency_workers=3,
)


class Bench:
    def __init__(self, plan: Plan, workers: int, reuse_rankings: bool, command: str) -> None:
        self.plan = plan
        self.workers = workers
        self.reuse_rankings = reuse_rankings
        self.command = command
        self.root = data_dir() / "bench-smoke" if plan.prefix else data_dir()
        self._splits: dict[str, list[Example]] = {}

    # -- paths and inputs ------------------------------------------------------------------

    def rankings_file(self, split: str) -> Path:
        return rankings_path(split, self.root)

    def models(self) -> Path:
        return models_dir(self.root)

    def runs(self) -> Path:
        return runs_dir(self.root)

    def split(self, name: str) -> list[Example]:
        if name not in self._splits:
            examples = load_split(name)  # type: ignore[arg-type]
            limit = self.plan.limits[name]
            self._splits[name] = examples[:limit] if limit else examples
        return self._splits[name]

    def rankings(self, name: str) -> dict[str, Ranking]:
        return load_rankings(self.rankings_file(name))

    def search_setup(self, uniform: bool = False) -> SearchSetup:
        models = self.models()
        return SearchSetup(
            None if uniform else models / "prior.txt",
            models / "ranker.txt",
            context=self.plan.train.context,
            max_steps=self.plan.train.max_steps,
        )

    def run_file(self, label: str) -> Path:
        """Per-question records of one test-split method (kept out of the repository)."""
        safe = label.replace(" ", "_").replace("+", "_")
        return self.runs() / f"{self.plan.prefix}test-{safe}.jsonl"

    def write(self, stage: str, config: dict[str, Any], metrics: dict[str, Any]) -> Path:
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"{self.plan.prefix}{stage}.json"
        body = {"stage": stage, **stamp(self.command), "config": config, "metrics": metrics}
        path.write_text(json.dumps(body, indent=1, default=str) + "\n", encoding="utf-8")
        print(f"[{stage}] wrote {path}", flush=True)
        return path

    def read(self, stage: str) -> dict[str, Any] | None:
        path = RESULTS / f"{self.plan.prefix}{stage}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    # -- stages ----------------------------------------------------------------------------

    def data(self) -> None:
        start = time.perf_counter()
        hashes_ok = verify_all(log=lambda _: None)
        full = {name: load_split(name) for name in SPLITS}  # type: ignore[misc]
        allow = load_allowlist()
        gold = {
            name: check_split(examples, allow if name == "train" else {}).summary()
            for name, examples in full.items()
        }
        lengths = Counter(
            len(p) if (p := e.gold_program()) is not None else 0
            for examples in full.values()
            for e in examples
        )
        total = sum(lengths.values())
        metrics = {
            "sha256_verified": hashes_ok,
            "questions": {name: len(ex) for name, ex in full.items()},
            "questions_total": total,
            "report_pages": {name: len({e.filename for e in ex}) for name, ex in full.items()},
            "gold_programs": gold,
            "leakage": leakage(full),  # type: ignore[arg-type]
            "program_steps": {str(k): v for k, v in sorted(lengths.items())},
            "share_within_4_steps": sum(v for k, v in lengths.items() if 1 <= k <= 4) / total,
            "seconds": time.perf_counter() - start,
        }
        self.write("data", {}, metrics)

    def retrieval(self) -> None:
        config = RetrieverConfig()
        idf = fit_idf(load_split("train"))
        save_idf(idf_path(self.root), idf)
        timings: dict[str, Any] = {}
        for name in SPLITS:
            path = self.rankings_file(name)
            examples = self.split(name)
            cached = self.reuse_rankings and path.is_file()
            if cached:
                have = self.rankings(name)
                cached = all(e.id in have for e in examples)
            if not cached:
                start = time.perf_counter()
                ranked = rank_examples(
                    examples, idf, config, self.workers, progress_printer(name, 500)
                )
                save_rankings(path, ranked)
                timings[name] = {"wall_seconds": time.perf_counter() - start}
            timings.setdefault(name, {"wall_seconds": None, "reused_cache": True})
            secs = sorted(r.seconds for r in self.rankings(name).values())
            timings[name]["per_question_p50"] = secs[len(secs) // 2]
            timings[name]["per_question_p95"] = secs[int(0.95 * (len(secs) - 1))]
        recall: dict[str, Any] = {}
        for name in ("dev", "test"):
            ranks = self.rankings(name)
            examples = [e for e in self.split(name) if e.gold_inds]
            recall[name] = {"questions_with_gold_facts": len(examples)}
            for label, transform in (("hybrid", lambda r: r), ("bm25", bm25_only)):
                for k in (1, 3, 5):
                    vals = [recall_at_k(transform(ranks[e.id]), e.gold_inds, k) for e in examples]
                    got = [v for v in vals if v is not None]
                    recall[name][f"{label}_recall@{k}"] = sum(got) / len(got)
                    recall[name][f"{label}_all_gold@{k}"] = sum(v == 1.0 for v in got) / len(got)
        self.write(
            "retrieval",
            {**asdict(config), "workers": self.workers, "idf_documents": idf.n_docs},
            {"recall": recall, "timing": timings},
        )

    def train(self) -> None:
        manifest = train(
            self.split("train"),
            self.rankings("train"),
            self.plan.train,
            self.models(),
            workers=self.workers,
        )
        sizes = {p.name: p.stat().st_size for p in self.models().glob("*.txt")}
        self.write(
            "train",
            asdict(self.plan.train) | {"workers": self.workers},
            {**{k: v for k, v in manifest.items() if k != "setup"}, "model_bytes": sizes},
        )

    def tune(self) -> None:
        dev = self.split("dev")
        ranks = self.rankings("dev")
        setup = self.search_setup()
        sims = self.plan.tune_simulations
        rows: list[dict[str, Any]] = []

        def evaluate(methods: list[Method]) -> None:
            out = run_split(dev, ranks, methods, setup, self.workers)
            for m in methods:
                row = {"label": m.label, **summarise(out[m.label])}
                if m.name == "mcts":
                    row.update(
                        c_puct=m.mcts.c_puct,
                        fpu_reduction=m.mcts.fpu_reduction,
                        prior_weight=m.prior_weight,
                    )
                rows.append(row)
                print(f"  {m.label}: {row['execution_accuracy']:.4f}", flush=True)

        reference = [Method("greedy"), Method("beam", width=max(self.plan.beam_widths))]
        grid = [
            Method(
                "mcts",
                mcts=MCTSConfig(simulations=sims, c_puct=c, fpu_reduction=f),
                tag=f"mcts-{sims} c={c} fpu={f} w=1.0",
            )
            for c, f in self.plan.tune_grid
        ]
        evaluate(reference + grid)
        mcts_rows = [r for r in rows if "c_puct" in r]
        best = max(mcts_rows, key=lambda r: r["execution_accuracy"])
        weights = [w for w in self.plan.prior_weights if w != 1.0]
        evaluate(
            [
                Method(
                    "mcts",
                    mcts=MCTSConfig(
                        simulations=sims, c_puct=best["c_puct"], fpu_reduction=best["fpu_reduction"]
                    ),
                    prior_weight=w,
                    tag=f"mcts-{sims} c={best['c_puct']} fpu={best['fpu_reduction']} w={w}",
                )
                for w in weights
            ]
        )
        mcts_rows = [r for r in rows if "c_puct" in r]
        best = max(mcts_rows, key=lambda r: r["execution_accuracy"])
        chosen = {k: best[k] for k in ("c_puct", "fpu_reduction", "prior_weight")}
        self.write(
            "tune",
            {"split": "dev", "simulations": sims, "questions": len(dev)},
            {"grid": rows, "chosen": chosen},
        )

    def chosen(self) -> dict[str, float]:
        tuned = self.read("tune")
        if tuned is None:
            print("no tune results; using defaults", flush=True)
            return {"c_puct": 1.5, "fpu_reduction": 0.0, "prior_weight": 1.0}
        return dict(tuned["metrics"]["chosen"])

    def mcts_method(self, simulations: int, trees: int = 1) -> Method:
        c = self.chosen()
        return Method(
            "mcts",
            mcts=MCTSConfig(
                simulations=simulations, c_puct=c["c_puct"], fpu_reduction=c["fpu_reduction"]
            ),
            prior_weight=c["prior_weight"],
            trees=trees,
        )

    def test(self) -> None:
        test = self.split("test")
        ranks = self.rankings("test")
        weight = self.chosen()["prior_weight"]
        methods = [Method("greedy")]
        methods += [Method("beam", width=w) for w in self.plan.beam_widths]
        methods += [
            Method("beam+ranker", width=w, prior_weight=weight) for w in self.plan.beam_widths
        ]
        methods += [Method("depth2+ranker")]
        methods += [self.mcts_method(n) for n in self.plan.simulations]
        start = time.perf_counter()
        out = run_split(test, ranks, methods, self.search_setup(), self.workers)
        wall = time.perf_counter() - start
        mid = self.plan.simulations[len(self.plan.simulations) // 2]
        uniform = replace(self.mcts_method(mid), tag=f"mcts-{mid} uniform prior")
        start = time.perf_counter()
        out |= run_split(test, ranks, [uniform], self.search_setup(uniform=True), self.workers)
        wall_uniform = time.perf_counter() - start
        self.runs().mkdir(parents=True, exist_ok=True)
        for label, records in out.items():
            with self.run_file(label).open("w", encoding="utf-8") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in records)
        self.write(
            "test",
            {
                "split": "test",
                "questions": len(test),
                "workers": self.workers,
                "tuned": self.chosen(),
                "methods": [m.label for m in methods] + [uniform.label],
            },
            {
                "methods": {label: summarise(records) for label, records in out.items()},
                "wall_seconds": wall,
                "wall_seconds_uniform_prior": wall_uniform,
            },
        )

    def latency(self) -> None:
        test = self.split("test")[: self.plan.latency_questions]
        ranks = self.rankings("test")
        sims = self.plan.latency_simulations
        results: dict[str, Any] = {}
        for workers in (1, self.plan.latency_workers):
            method = self.mcts_method(sims, trees=workers)
            records = run_latency(test, ranks, method, self.search_setup(), workers)
            results[f"{workers} worker{'s' if workers > 1 else ''}"] = {
                "trees": workers,
                "simulations_per_tree": split_budget(sims, workers),
                **summarise_latency(records),
            }
        self.write(
            "latency",
            {"split": "test", "questions": len(test), "simulations": sims},
            results,
        )

    def trace(self) -> None:
        sims = self.plan.simulations[len(self.plan.simulations) // 2]
        method = self.mcts_method(sims)
        runs = self.run_file(method.label)
        test = {e.id: e for e in self.split("test")}
        ranks = self.rankings("test")
        prior, ranker = self.search_setup().load()
        candidates = [json.loads(line) for line in runs.read_text(encoding="utf-8").splitlines()]
        picked = None
        for rec in candidates:
            ex = test[rec["id"]]
            program = ex.gold_program()
            if rec["exe_correct"] and program is not None and len(program) == 2:
                picked = rec["id"]
                break
        if picked is None:
            picked = candidates[0]["id"]
        item = make_item(test[picked], ranks[picked], self.plan.train.context, 4)
        record = trace_record(item, solve(item, method, prior, ranker, with_trace=True))
        sketch = mermaid(record, max_children=3, max_depth=2)
        self.write(
            "trace",
            {
                "question": picked,
                "method": method.label,
                "rule": "first test question in "
                "file order whose gold program has two steps and that this method answers",
            },
            {"record": record, "mermaid": sketch},
        )


def summarise_latency(records: list[dict[str, Any]]) -> dict[str, Any]:
    secs = sorted(r["seconds"] for r in records)
    n = len(records)

    def pct(q: float) -> float:
        return secs[min(n - 1, round(q * (n - 1)))]

    return {
        "n": n,
        "seconds_p50": pct(0.5),
        "seconds_p95": pct(0.95),
        "seconds_mean": sum(secs) / n,
        "execution_accuracy": sum(r["exe_correct"] for r in records) / n,
        "program_accuracy": sum(r["prog_correct"] for r in records) / n,
        "candidate_coverage": sum(r["candidates_correct"] for r in records) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true", help="tiny subsets, smoke-*.json")
    parser.add_argument(
        "--reuse-rankings",
        action="store_true",
        help="keep cached fact rankings instead of recomputing them",
    )
    args = parser.parse_args()
    command = "uv run python bench/run.py " + " ".join(sys.argv[1:])
    bench = Bench(SMOKE if args.smoke else FULL, args.workers, args.reuse_rankings, command)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        parser.error(f"unknown stages: {sorted(unknown)}")
    total = time.perf_counter()
    for stage in STAGES:
        if stage in stages:
            start = time.perf_counter()
            print(f"[{stage}] starting", flush=True)
            getattr(bench, stage)()
            print(f"[{stage}] done in {time.perf_counter() - start:.0f} s", flush=True)
    print(f"all stages done in {time.perf_counter() - total:.0f} s", flush=True)


if __name__ == "__main__":
    main()
