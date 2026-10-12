"""Hybrid fact retrieval: BM25 picks a pool, the cross-encoder reorders it.

The final score inside the pool is ``z(cross-encoder) + w * z(BM25)``, z-scored per
question; facts outside the pool follow in BM25 order. Rankings are cached per split
as JSON lines, because the cross-encoder dominates the cost (~30 ms per pair on the
reference CPU) and every later stage reads the same ranking.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from finqa_search.dataset import Example, report_of
from finqa_search.retrieval.bm25 import BM25, IdfTable
from finqa_search.retrieval.cross_encoder import CrossEncoder


class Ranking(BaseModel):
    """Facts of one example, best first, with the scores that ordered them."""

    id: str
    fact_ids: list[str]
    scores: list[float]
    bm25: list[float]
    cross: list[float | None]
    seconds: float = 0.0

    def rank_of(self) -> dict[str, int]:
        return {f: i for i, f in enumerate(self.fact_ids)}

    def top(self, k: int) -> list[str]:
        return self.fact_ids[:k]


@dataclass
class RetrieverConfig:
    pool: int = 12
    bm25_weight: float = 0.3
    use_cross_encoder: bool = True


def fit_idf(train: Iterable[Example]) -> IdfTable:
    """IDF over the facts of each distinct training page (pages repeat across questions)."""
    seen: set[str] = set()
    docs: list[str] = []
    for ex in train:
        page = report_of(ex.id)
        if page in seen:
            continue
        seen.add(page)
        docs.extend(f.text for f in ex.facts())
    return IdfTable.fit(docs)


def _z(x: np.ndarray) -> np.ndarray:
    sd = float(x.std())
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


class FactRetriever:
    def __init__(
        self,
        idf: IdfTable,
        config: RetrieverConfig | None = None,
        cross_encoder: CrossEncoder | None = None,
    ) -> None:
        self.config = config or RetrieverConfig()
        self.bm25 = BM25(idf)
        self.cross_encoder = cross_encoder if self.config.use_cross_encoder else None

    def rank(self, example: Example) -> Ranking:
        start = time.perf_counter()
        facts = example.facts()
        texts = [f.text for f in facts]
        bm = np.asarray(self.bm25.score(example.question, texts), dtype=np.float64)
        order = np.argsort(-bm, kind="stable")
        cross: list[float | None] = [None] * len(facts)
        if self.cross_encoder is None or not facts:
            final = order
            scores = bm[order]
        else:
            pool = order[: self.config.pool]
            ce = self.cross_encoder.score(example.question, [texts[i] for i in pool])
            ce = ce.astype(np.float64)
            for i, s in zip(pool, ce, strict=True):
                cross[int(i)] = float(s)
            fused = _z(ce) + self.config.bm25_weight * _z(bm[pool])
            inner = np.argsort(-fused, kind="stable")
            rest = order[self.config.pool :]
            final = np.concatenate([pool[inner], rest])
            # Facts outside the pool rank below every pooled fact.
            floor = float(fused.min()) - 1.0 if len(fused) else 0.0
            scores = np.concatenate([fused[inner], floor - np.arange(1, len(rest) + 1)])
        return Ranking(
            id=example.id,
            fact_ids=[facts[i].id for i in final],
            scores=[float(s) for s in scores],
            bm25=[float(bm[i]) for i in final],
            cross=[cross[int(i)] for i in final],
            seconds=time.perf_counter() - start,
        )


def recall_at_k(ranking: Ranking, gold: Iterable[str], k: int) -> float | None:
    gold_set = set(gold)
    if not gold_set:
        return None
    return len(set(ranking.fact_ids[:k]) & gold_set) / len(gold_set)


def bm25_only(ranking: Ranking) -> Ranking:
    """The same facts re-sorted by BM25 alone, for the retrieval ablation."""
    order = np.argsort(-np.asarray(ranking.bm25), kind="stable")
    return Ranking(
        id=ranking.id,
        fact_ids=[ranking.fact_ids[i] for i in order],
        scores=[ranking.bm25[i] for i in order],
        bm25=[ranking.bm25[i] for i in order],
        cross=[ranking.cross[i] for i in order],
    )


def save_rankings(path: Path, rankings: Iterable[Ranking]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rankings:
            fh.write(r.model_dump_json() + "\n")
    tmp.replace(path)


def load_rankings(path: Path) -> dict[str, Ranking]:
    out: dict[str, Ranking] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = Ranking.model_validate_json(line)
                out[r.id] = r
    return out


def save_idf(path: Path, idf: IdfTable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n_docs": idf.n_docs, "df": idf.df}), encoding="utf-8")


def load_idf(path: Path) -> IdfTable:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return IdfTable(n_docs=int(raw["n_docs"]), df={k: int(v) for k, v in raw["df"].items()})
