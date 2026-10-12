"""Fit the prior and the ranker on the training split.

1. In worker processes, build each question's context and gold trajectory and emit the
   prior's ranking groups.
2. Fit the full prior, plus one prior per fold of a 3-fold split by report page.
3. In worker processes, generate ranker candidates for each question with the prior of
   the fold that excludes it (beam, ancestral samples, uniform depth-2 draws, gold) and
   label them by execution.
4. Fit one ranker per fold on those candidates, run MCTS on each question with the
   models of the fold that excludes it, and add every program it scored.
5. Fit the final ranker on both rounds.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from finqa_search.context import ContextConfig
from finqa_search.dataset import Example, report_of
from finqa_search.prior import LightGBMPrior
from finqa_search.ranker import ProgramRanker
from finqa_search.retrieval.retriever import Ranking
from finqa_search.training import (
    PriorConfig,
    RankerConfig,
    fit_prior_rows,
    fit_ranker,
    make_item,
    parallel_map,
    prior_rows,
    ranker_rows,
    search_rows,
)


@dataclass(frozen=True)
class TrainSetup:
    context: ContextConfig
    max_steps: int
    prior: PriorConfig
    ranker: RankerConfig


def fold_of(example_id: str, folds: int) -> int:
    """Stable fold by report page, so questions on one page never straddle folds."""
    digest = hashlib.sha256(report_of(example_id).encode("utf-8")).digest()
    return digest[0] % folds


_W: dict[str, Any] = {}


def _init_prior_job(setup: TrainSetup) -> None:
    _W["setup"] = setup


def _prior_job(job: tuple[Example, Ranking]) -> tuple[np.ndarray, np.ndarray, list[int], bool]:
    example, ranking = job
    setup: TrainSetup = _W["setup"]
    item = make_item(example, ranking, setup.context, setup.max_steps)
    if item.gold is None:
        return np.zeros((0, 0), np.float32), np.zeros(0), [], False
    seed = int.from_bytes(hashlib.sha256(example.id.encode()).digest()[:4], "little")
    X, y, g = prior_rows(item, np.random.default_rng(seed), setup.prior.negatives)
    return X, y, g, True


def _init_ranker_job(setup: TrainSetup, prior_paths: list[str]) -> None:
    _W["setup"] = setup
    _W["priors"] = [LightGBMPrior.load(Path(p)) for p in prior_paths]


def _ranker_job(job: tuple[Example, Ranking, int]) -> tuple[np.ndarray, np.ndarray]:
    example, ranking, fold = job
    setup: TrainSetup = _W["setup"]
    item = make_item(example, ranking, setup.context, setup.max_steps)
    seed = int.from_bytes(hashlib.sha256(example.id.encode()).digest()[4:8], "little")
    return ranker_rows(item, _W["priors"][fold], setup.ranker, seed)


def _init_search_job(setup: TrainSetup, prior_paths: list[str], ranker_paths: list[str]) -> None:
    _W["setup"] = setup
    _W["priors"] = [LightGBMPrior.load(Path(p)) for p in prior_paths]
    _W["rankers"] = [ProgramRanker.load(Path(p)) for p in ranker_paths]


def _search_job(job: tuple[Example, Ranking, int]) -> tuple[np.ndarray, np.ndarray]:
    example, ranking, fold = job
    setup: TrainSetup = _W["setup"]
    item = make_item(example, ranking, setup.context, setup.max_steps)
    return search_rows(
        item, _W["priors"][fold], _W["rankers"][fold], setup.ranker.search_simulations
    )


def _stack(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    used = [p for p in parts if len(p[1])]
    return np.vstack([p[0] for p in used]), np.concatenate([p[1] for p in used])


def train(
    examples: list[Example],
    rankings: dict[str, Ranking],
    setup: TrainSetup,
    out_dir: Path,
    workers: int = 6,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    jobs = [(e, rankings[e.id]) for e in examples]
    rows = parallel_map(_prior_job, jobs, workers, _init_prior_job, (setup,))
    reachable = sum(1 for r in rows if r[3])
    log(f"prior rows built for {reachable}/{len(examples)} reachable questions")
    folds = setup.ranker.folds
    fold_ids = [fold_of(e.id, folds) for e in examples]
    t_rows = time.perf_counter() - t0

    t1 = time.perf_counter()
    full = fit_prior_rows(rows, setup.prior, threads=workers)
    full.save(out_dir / "prior.txt")
    fold_paths: list[str] = []
    for k in range(folds):
        part = [r for r, f in zip(rows, fold_ids, strict=True) if f != k]
        model = fit_prior_rows(part, setup.prior, threads=workers)
        path = out_dir / f"prior-fold{k}.txt"
        model.save(path)
        fold_paths.append(str(path))
    t_prior = time.perf_counter() - t1
    log(f"priors fitted in {t_prior:.0f} s")
    del rows

    t2 = time.perf_counter()
    rjobs = [(e, rankings[e.id], f) for e, f in zip(examples, fold_ids, strict=True)]
    parts = parallel_map(_ranker_job, rjobs, workers, _init_ranker_job, (setup, fold_paths))
    X, y = _stack(parts)
    t_cands = time.perf_counter() - t2
    log(f"ranker candidates: {len(y)} programs, {y.mean():.3f} positive, {t_cands:.0f} s")

    t_search = 0.0
    n_first = len(y)
    temp_paths = list(fold_paths)
    if setup.ranker.search_simulations:
        t4 = time.perf_counter()
        ranker_paths: list[str] = []
        for k in range(folds):
            Xk, yk = _stack([p for p, f in zip(parts, fold_ids, strict=True) if f != k])
            path = out_dir / f"ranker-fold{k}.txt"
            fit_ranker(Xk, yk, setup.ranker, threads=workers).save(path)
            ranker_paths.append(str(path))
        temp_paths += ranker_paths
        found = parallel_map(
            _search_job, rjobs, workers, _init_search_job, (setup, fold_paths, ranker_paths)
        )
        Xs, ys = _stack(found)
        X, y = np.vstack([X, Xs]), np.concatenate([y, ys])
        t_search = time.perf_counter() - t4
        log(f"search candidates: {len(ys)} programs, {ys.mean():.3f} positive, {t_search:.0f} s")
    del parts

    t3 = time.perf_counter()
    ranker = fit_ranker(X, y, setup.ranker, threads=workers)
    ranker.save(out_dir / "ranker.txt")
    t_ranker = time.perf_counter() - t3
    for p in temp_paths:
        Path(p).unlink(missing_ok=True)

    manifest = {
        "train_questions": len(examples),
        "gold_reachable": reachable,
        "ranker_programs": len(y),
        "ranker_programs_from_search": len(y) - n_first,
        "ranker_positive_rate": float(y.mean()),
        "seconds": {
            "prior_rows": t_rows,
            "prior_fit": t_prior,
            "ranker_candidates": t_cands,
            "ranker_search_candidates": t_search,
            "ranker_fit": t_ranker,
            "total": time.perf_counter() - t0,
        },
        "setup": asdict(setup),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
