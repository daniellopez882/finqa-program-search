from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from finqa_search.dataset import Example
from finqa_search.retrieval.bm25 import BM25, IdfTable, stem, tokenize
from finqa_search.retrieval.retriever import (
    FactRetriever,
    RetrieverConfig,
    bm25_only,
    fit_idf,
    load_idf,
    load_rankings,
    recall_at_k,
    save_idf,
    save_rankings,
)


def make_example(question: str) -> Example:
    return Example(
        id="ACME/2010/page_1.pdf-1",
        filename="ACME/2010/page_1.pdf",
        question=question,
        pre_text=[
            "the company sells widgets .",
            "net revenue increased in 2010 due to higher widget volume .",
        ],
        post_text=["interest expense was flat ."],
        table=[
            ["", "2010", "2009"],
            ["net revenue", "$ 120", "$ 100"],
            ["interest expense", "5", "5"],
        ],
        program="subtract(120, 100), divide(#0, 100)",
        exe_ans=0.2,
        gold_inds={"table_1": "the 2010 of net revenue is $ 120 ;"},
    )


def test_tokenize_lowercases_stems_and_drops_stopwords() -> None:
    assert tokenize("The Revenues of 2009 were $5.8 billion") == [
        "revenue",
        "2009",
        "5.8",
        "billion",
    ]
    assert stem("companies") == "company"
    assert stem("process") == "process"
    assert stem("bus") == "bus"


def test_idf_penalises_common_terms() -> None:
    idf = IdfTable.fit(["revenue grew", "revenue fell", "interest paid"])
    assert idf.idf("interest") > idf.idf("revenue") > 0
    assert idf.idf("unseen") > idf.idf("interest")


def test_bm25_prefers_the_matching_document() -> None:
    idf = IdfTable.fit(["net revenue", "interest expense", "widget volume", "net income"])
    scores = BM25(idf).score(
        "what was the change in interest expense", ["net revenue", "interest expense"]
    )
    assert scores[1] > scores[0]
    assert BM25(idf).score("anything", []) == []


def test_facts_cover_sentences_and_table_rows() -> None:
    ex = make_example("q")
    ids = [f.id for f in ex.facts()]
    assert ids == ["text_0", "text_1", "text_2", "table_1", "table_2"]
    row = next(f for f in ex.facts() if f.id == "table_1")
    assert row.text == "the 2010 of net revenue is $ 120 ; the 2009 of net revenue is $ 100 ;"


class FakeCrossEncoder:
    """Scores a passage by whether it mentions a keyword, standing in for the ONNX model."""

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword
        self.calls = 0

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        self.calls += 1
        return np.array([5.0 if self.keyword in p else 0.0 for p in passages], dtype=np.float32)


def test_bm25_only_ranking_puts_lexical_matches_first() -> None:
    ex = make_example("what was the percent change in net revenue from 2009 to 2010?")
    idf = fit_idf([ex])
    ranking = FactRetriever(idf, RetrieverConfig(use_cross_encoder=False)).rank(ex)
    assert ranking.fact_ids[0] in {"table_1", "text_1"}
    assert all(c is None for c in ranking.cross)
    assert sorted(ranking.fact_ids) == sorted(f.id for f in ex.facts())


def test_cross_encoder_reorders_the_pool_only() -> None:
    ex = make_example("what was the percent change in net revenue from 2009 to 2010?")
    idf = fit_idf([ex])
    ce = FakeCrossEncoder("interest")
    config = RetrieverConfig(pool=3, bm25_weight=0.0)
    ranking = FactRetriever(idf, config, ce).rank(ex)  # type: ignore[arg-type]
    plain = FactRetriever(idf, RetrieverConfig(use_cross_encoder=False)).rank(ex)
    pool = set(plain.fact_ids[:3])
    assert set(ranking.fact_ids[:3]) == pool
    assert ranking.fact_ids[3:] == plain.fact_ids[3:]
    if "table_2" in pool:
        assert ranking.fact_ids[0] == "table_2"
    assert ce.calls == 1
    assert sum(c is not None for c in ranking.cross) == 3
    assert ranking.scores == sorted(ranking.scores, reverse=True)
    resorted = bm25_only(ranking)
    assert resorted.fact_ids == plain.fact_ids


def test_recall_at_k_against_gold_indices() -> None:
    ex = make_example("net revenue change?")
    ranking = FactRetriever(fit_idf([ex]), RetrieverConfig(use_cross_encoder=False)).rank(ex)
    k = ranking.fact_ids.index("table_1") + 1
    assert recall_at_k(ranking, ex.gold_inds, k) == 1.0
    assert recall_at_k(ranking, ex.gold_inds, k - 1) == 0.0
    assert recall_at_k(ranking, {}, 3) is None


def test_rankings_and_idf_round_trip(tmp_path: Path) -> None:
    ex = make_example("net revenue change?")
    idf = fit_idf([ex])
    ranking = FactRetriever(idf, RetrieverConfig(use_cross_encoder=False)).rank(ex)
    save_rankings(tmp_path / "r.jsonl", [ranking])
    assert load_rankings(tmp_path / "r.jsonl")[ex.id] == ranking
    save_idf(tmp_path / "idf.json", idf)
    again = load_idf(tmp_path / "idf.json")
    assert again.idf("revenue") == pytest.approx(idf.idf("revenue"))
