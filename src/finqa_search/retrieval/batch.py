"""Rank the facts of a whole split in worker processes, one cross-encoder per worker."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from typing import Any

from finqa_search.dataset import Example
from finqa_search.retrieval.bm25 import IdfTable
from finqa_search.retrieval.cross_encoder import CrossEncoder
from finqa_search.retrieval.retriever import FactRetriever, Ranking, RetrieverConfig

_RETRIEVER: FactRetriever | None = None


def _init(idf: IdfTable, config: RetrieverConfig) -> None:
    global _RETRIEVER
    ce = CrossEncoder(threads=1) if config.use_cross_encoder else None
    _RETRIEVER = FactRetriever(idf, config, ce)


def _rank(example: Example) -> Ranking:
    assert _RETRIEVER is not None
    return _RETRIEVER.rank(example)


def rank_examples(
    examples: list[Example],
    idf: IdfTable,
    config: RetrieverConfig,
    workers: int = 1,
    progress: Callable[[int, int], Any] | None = None,
) -> list[Ranking]:
    out: list[Ranking] = []
    if workers <= 1:
        _init(idf, config)
        for i, ex in enumerate(examples):
            out.append(_rank(ex))
            if progress:
                progress(i + 1, len(examples))
        return out
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(idf, config)) as pool:
        for i, r in enumerate(pool.imap(_rank, examples, chunksize=8)):
            out.append(r)
            if progress:
                progress(i + 1, len(examples))
    return out
