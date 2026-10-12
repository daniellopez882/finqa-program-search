"""Feature layout shared by the action prior and the program ranker.

Every argument a step can take (a number from the report, a constant, an earlier step
result or a table row) is described by the same ``ARG_FEATURES`` vector, so a step is
"op + two argument vectors + how they relate + what the step evaluates to". Values that
do not apply are NaN, which LightGBM treats as missing.
"""

from __future__ import annotations

import math
import re
from typing import Final

import numpy as np

from finqa_search.dsl import ARITH_OPS, TABLE_OPS

OPS: Final = (*ARITH_OPS, *TABLE_OPS, "stop")
OP_ID: Final = {op: i for i, op in enumerate(OPS)}
STOP: Final = OP_ID["stop"]

KIND_NUMBER, KIND_CONST, KIND_REF, KIND_ROW = 0, 1, 2, 3

ARG_FEATURES: Final = (
    "kind",
    "const_log",
    "fact_rank",
    "fact_gap",
    "in_table",
    "in_text",
    "in_question",
    "n_mentions",
    "log_abs",
    "negative",
    "percent",
    "variant",
    "year_like",
    "year_in_q",
    "year_is_q_max",
    "year_is_q_min",
    "label_overlap",
    "context_overlap",
    "header_overlap",
    "col",
    "unit_log",
    "ref_op",
    "ref_is_last",
    "ref_size",
    "q_pos",
)
N_ARG: Final = len(ARG_FEATURES)
A = {name: i for i, name in enumerate(ARG_FEATURES)}

VARIANT_ID: Final = {"": 0.0, "pct": 1.0, "abs": 2.0, "neg": 3.0}

QUESTION_PATTERNS: Final[dict[str, str]] = {
    "q_percent": r"\bpercent|\bpercentage|%|\brate\b",
    "q_change": r"\bchange|\bincrease|\bdecrease|\bgrowth|\bgrow|\bdecline|\brise|\bfell\b|\bdrop",
    "q_ratio": r"\bratio|\bproportion|\bportion|\bfraction|\bshare of|\bpercent of|\bas a percent",
    "q_average": r"\baverage|\bmean\b",
    "q_total": r"\btotal|\bsum\b|\bcombined|\baggregate|\btogether|\bcumulative",
    "q_difference": r"\bdifference|\bmore than|\bless than|\bhigher|\blower|\bnet change",
    "q_times": r"\btimes\b|\bmultiple",
    "q_yes_no": r"^\s*(is|was|were|did|does|do|are|has|have|will|can)\b",
    "q_compare": r"\bgreater|\bexceed|\blarger|\bbigger|\bmore\b|\bless\b|\bsmaller",
    "q_million": r"\bmillion",
    "q_billion": r"\bbillion",
    "q_thousand": r"\bthousand",
    "q_per_share": r"\bper share|\beach share",
    "q_compound": r"\bcagr|\bcompound|\bannual(?:ized)? (?:growth|rate)",
    "q_years_word": r"\bin years|\bhow many years|\bhow long",
    "q_how_many": r"\bhow many|\bhow much",
    "q_if": r"\bif\b|\bassum",
    "q_remaining": r"\bremain|\bwithout|\bexclud|\bexcept|\bother than",
}
Q_FEATURES: Final = (*QUESTION_PATTERNS, "q_n_years", "q_has_number", "q_len")
_Q_RE = {k: re.compile(v) for k, v in QUESTION_PATTERNS.items()}

STATE_FEATURES: Final = ("op", "depth", "n_open", "prev_op", "last_log_abs")
PAIR_FEATURES: Final = (
    "same_row",
    "same_col",
    "same_fact",
    "year_order",
    "log_ratio",
    "a_gt_b",
    "q_order",
)
RESULT_FEATURES: Final = ("res_log_abs", "res_negative", "res_below_one", "res_integer")

ACTION_FEATURES: Final = (
    *STATE_FEATURES,
    *(f"a_{n}" for n in ARG_FEATURES),
    *(f"b_{n}" for n in ARG_FEATURES),
    *PAIR_FEATURES,
    *RESULT_FEATURES,
    *Q_FEATURES,
)
ACTION_CATEGORICAL: Final = ("op", "prev_op", "a_kind", "b_kind", "a_ref_op", "b_ref_op")


def question_features(question: str, years: list[int]) -> np.ndarray:
    low = question.lower()
    flags = [1.0 if rx.search(low) else 0.0 for rx in _Q_RE.values()]
    has_number = 1.0 if re.search(r"\d", re.sub(r"\b(19|20)\d\d\b", " ", low)) else 0.0
    return np.array([*flags, float(len(set(years))), has_number, float(len(low.split()))])


def question_position(anchor: tuple[str, ...], q_tokens: list[str]) -> float:
    """Where (0 = start, 1 = end) the question first uses a word that names a number.

    For "ratio of capital to net income", the number labelled "capital" sits earlier
    than the one labelled "net income"; comparing two positions gives operand order.
    """
    names = set(anchor)
    for i, tok in enumerate(q_tokens):
        if tok in names:
            return i / max(len(q_tokens) - 1, 1)
    return math.nan


def log_abs(x: np.ndarray | float) -> np.ndarray:
    return np.log10(np.abs(np.asarray(x, dtype=np.float64)) + 1e-9)


def scalar_log_abs(x: float) -> float:
    return math.log10(abs(x) + 1e-9)


def result_features(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    out = np.empty((len(v), len(RESULT_FEATURES)))
    out[:, 0] = log_abs(v)
    out[:, 1] = v < 0
    out[:, 2] = np.abs(v) < 1
    out[:, 3] = np.isclose(v, np.round(v))
    return out
