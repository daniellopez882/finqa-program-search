"""Everything the search needs about one question, computed once before searching.

A context holds the leaf arguments (numbers from the top-ranked facts plus the typed
constants), the table rows the table operations may aggregate, and one feature vector
per argument. Nothing here depends on the partial program; that part lives in
``space.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from finqa_search.dataset import Example
from finqa_search.dsl import CONSTANTS, Const, Num, ProgramError, table_row_values, value_key
from finqa_search.features import (
    KIND_CONST,
    KIND_NUMBER,
    KIND_ROW,
    N_ARG,
    VARIANT_ID,
    A,
    question_features,
    question_position,
    scalar_log_abs,
)
from finqa_search.numbers import Candidate, Mention, example_mentions, merge, years_in
from finqa_search.retrieval.bm25 import tokenize
from finqa_search.retrieval.retriever import Ranking

# Which (operation, slot) each constant may fill. The table is read off the training
# programs: averages divide by small counts, units scale by 1000 or a million, growth
# subtracts 1, and "100" appears on both sides of subtract and divide.
CONSTANT_SLOTS: dict[str, frozenset[tuple[str, int]]] = {
    "const_100": frozenset(
        {("subtract", 1), ("divide", 1), ("multiply", 1), ("divide", 0), ("subtract", 0)}
    ),
    "const_1000": frozenset({("multiply", 1), ("divide", 1)}),
    "const_1000000": frozenset({("multiply", 1), ("divide", 1)}),
    "const_1": frozenset({("subtract", 1), ("subtract", 0), ("add", 1)}),
    "const_m1": frozenset({("multiply", 1)}),
    "const_2": frozenset({("divide", 1), ("multiply", 1), ("add", 1)}),
    "const_3": frozenset({("divide", 1), ("add", 1)}),
    "const_4": frozenset({("divide", 1), ("add", 1)}),
    "const_5": frozenset({("divide", 1), ("add", 1)}),
}


@dataclass(frozen=True)
class ContextConfig:
    top_facts: int = 5
    max_numbers: int = 18
    constants: tuple[str, ...] = tuple(CONSTANT_SLOTS)
    max_rows: int = 6


@dataclass(frozen=True, slots=True)
class Leaf:
    arg: Num | Const
    value: float
    mention: Mention | None = None


@dataclass(frozen=True, slots=True)
class RowArg:
    label: str
    values: tuple[float, ...]
    row: int


@dataclass
class QuestionContext:
    example_id: str
    question: str
    table: list[list[str]]
    leaves: list[Leaf]
    rows: list[RowArg]
    leaf_feats: np.ndarray
    row_feats: np.ndarray
    leaf_values: np.ndarray
    leaf_row: np.ndarray
    leaf_col: np.ndarray
    leaf_fact: np.ndarray
    leaf_year: np.ndarray
    q_feats: np.ndarray
    q_years: list[int]
    fact_rank: dict[str, int] = field(default_factory=dict)

    @property
    def n_leaves(self) -> int:
        return len(self.leaves)

    def leaf_index(self) -> dict[str, int]:
        """value key -> leaf, used to map a gold program onto this context."""
        return {value_key(leaf.value): i for i, leaf in enumerate(self.leaves)}


def _overlap(part: set[str], whole: set[str]) -> float:
    return len(part & whole) / len(part) if part else math.nan


def _representative(cand: Candidate, rank: dict[str, int]) -> Mention:
    """The mention that describes a candidate: the best-ranked one outside the question."""
    pool = [m for m in cand.mentions if m.kind != "question"] or cand.mentions
    return min(pool, key=lambda m: (rank.get(m.fact_id, 10_000), m.variant != "", m.position))


def _mention_rank(m: Mention, rank: dict[str, int], table_best: int) -> int:
    if m.kind == "question":
        return -1
    if m.fact_id == "table_0":
        return table_best
    return rank.get(m.fact_id, 10_000)


def build_context(
    example: Example, ranking: Ranking, config: ContextConfig | None = None
) -> QuestionContext:
    cfg = config or ContextConfig()
    rank = ranking.rank_of()
    score = dict(zip(ranking.fact_ids, ranking.scores, strict=True))
    top_score = ranking.scores[0] if ranking.scores else 0.0
    top = set(ranking.top(cfg.top_facts))
    table_ranks = [r for f, r in rank.items() if f.startswith("table_")]
    table_best = min(table_ranks) if table_ranks else 10_000

    q_list = [t for t in tokenize(example.question) if not t[0].isdigit()]
    q_tokens = set(q_list)
    q_years = sorted(set(years_in(example.question)))

    mentions = example_mentions(example, top)
    cands = merge(mentions)

    def order_key(c: Candidate) -> tuple[int, bool, int]:
        rep = _representative(c, rank)
        best = min(_mention_rank(m, rank, table_best) for m in c.mentions)
        return (best, rep.variant != "", rep.position)

    cands.sort(key=order_key)
    cands = cands[: cfg.max_numbers]

    leaves: list[Leaf] = []
    rows_feats: list[np.ndarray] = []
    seen_values = set()
    for c in cands:
        rep = _representative(c, rank)
        leaves.append(Leaf(Num(rep.text, c.value), c.value, rep))
        seen_values.add(value_key(c.value))
    leaves.extend(
        Leaf(Const(name), CONSTANTS[name])
        for name in cfg.constants
        if value_key(CONSTANTS[name]) not in seen_values
    )

    n = len(leaves)
    feats = np.full((n, N_ARG), np.nan)
    leaf_row = np.full(n, -1, dtype=np.int64)
    leaf_col = np.full(n, -1, dtype=np.int64)
    leaf_fact = np.full(n, -1, dtype=np.int64)
    leaf_year = np.full(n, np.nan)
    cand_by_key = {value_key(c.value): c for c in cands}
    for i, leaf in enumerate(leaves):
        f = feats[i]
        f[A["log_abs"]] = scalar_log_abs(leaf.value)
        f[A["negative"]] = float(leaf.value < 0)
        if isinstance(leaf.arg, Const):
            f[A["kind"]] = KIND_CONST
            f[A["const_log"]] = scalar_log_abs(leaf.value)
            continue
        cand = cand_by_key[value_key(leaf.value)]
        rep = leaf.mention
        assert rep is not None
        ranks = [_mention_rank(m, rank, table_best) for m in cand.mentions]
        r = _mention_rank(rep, rank, table_best)
        f[A["kind"]] = KIND_NUMBER
        f[A["fact_rank"]] = min(ranks)
        f[A["fact_gap"]] = top_score - score.get(rep.fact_id, top_score) if r >= 0 else 0.0
        f[A["in_table"]] = float(any(m.kind == "table" for m in cand.mentions))
        f[A["in_text"]] = float(any(m.kind == "text" for m in cand.mentions))
        f[A["in_question"]] = float(any(m.kind == "question" for m in cand.mentions))
        f[A["n_mentions"]] = len(cand.mentions)
        f[A["percent"]] = float(any(m.percent for m in cand.mentions))
        f[A["variant"]] = VARIANT_ID[rep.variant]
        f[A["year_like"]] = float(
            leaf.value.is_integer() and 1950 <= leaf.value <= 2049 and rep.variant == ""
        )
        if rep.year is not None:
            leaf_year[i] = rep.year
            f[A["year_in_q"]] = float(rep.year in q_years)
            if q_years:
                f[A["year_is_q_max"]] = float(rep.year == q_years[-1])
                f[A["year_is_q_min"]] = float(rep.year == q_years[0])
        if rep.kind == "table":
            label = set(tokenize(rep.row_label))
            f[A["label_overlap"]] = _overlap(label, q_tokens)
            f[A["header_overlap"]] = _overlap(set(tokenize(rep.col_header)), q_tokens)
            f[A["col"]] = rep.col
            leaf_row[i] = rep.row
            leaf_col[i] = rep.col
        f[A["context_overlap"]] = _overlap(q_tokens, set(rep.context))
        f[A["q_pos"]] = question_position(rep.anchor, q_list)
        f[A["unit_log"]] = math.log10(rep.unit)
        leaf_fact[i] = rank.get(rep.fact_id, -1) if rep.kind != "question" else -2

    rows: list[RowArg] = []
    table_facts = [f for f in ranking.fact_ids[: cfg.top_facts] if f.startswith("table_")]
    labels_done: set[str] = set()
    for fid in table_facts:
        r_idx = int(fid.split("_")[1])
        if r_idx >= len(example.table) or not example.table[r_idx]:
            continue
        label = example.table[r_idx][0]
        if label in labels_done:
            continue
        try:
            values = table_row_values(example.table, label)
        except ProgramError:
            continue
        labels_done.add(label)
        rows.append(RowArg(label, tuple(values), r_idx))
        rf = np.full(N_ARG, np.nan)
        rf[A["kind"]] = KIND_ROW
        rf[A["fact_rank"]] = rank.get(fid, 10_000)
        rf[A["fact_gap"]] = top_score - score.get(fid, top_score)
        rf[A["in_table"]] = 1.0
        rf[A["n_mentions"]] = len(values)
        rf[A["log_abs"]] = scalar_log_abs(sum(values) / len(values))
        rf[A["label_overlap"]] = _overlap(set(tokenize(label)), q_tokens)
        rf[A["context_overlap"]] = _overlap(q_tokens, set(tokenize(label)))
        rf[A["q_pos"]] = question_position(tuple(tokenize(label)), q_list)
        rows_feats.append(rf)
        if len(rows) >= cfg.max_rows:
            break

    return QuestionContext(
        example_id=example.id,
        question=example.question,
        table=example.table,
        leaves=leaves,
        rows=rows,
        leaf_feats=feats,
        row_feats=np.array(rows_feats) if rows_feats else np.zeros((0, N_ARG)),
        leaf_values=np.array([leaf.value for leaf in leaves], dtype=np.float64),
        leaf_row=leaf_row,
        leaf_col=leaf_col,
        leaf_fact=leaf_fact,
        leaf_year=leaf_year,
        q_feats=question_features(example.question, q_years),
        q_years=q_years,
        fact_rank=rank,
    )
