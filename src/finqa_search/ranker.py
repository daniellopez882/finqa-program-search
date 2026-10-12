"""The program ranker: P(a finished program answers the question), from LightGBM.

It sees only the finished program and what grounds it (which numbers it used, where they
came from, what it evaluates to), never how a search reached it, so every search method
is judged by the same verifier. Features are computed for whole batches of programs at
once because the exhaustive depth-2 baseline scores a few hundred thousand per question.

Programs are passed as padded arrays: ``steps[n, S, 3]`` of (op, a, b) indices in the
``ActionSpace`` convention, ``values[n, S]`` of step results and ``lengths[n]``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import lightgbm as lgb
import numpy as np

from finqa_search.dsl import CONSTANTS
from finqa_search.features import OP_ID, Q_FEATURES, STOP, A, log_abs

if TYPE_CHECKING:
    from finqa_search.context import QuestionContext

_TABLE_OPS = np.array([OP_ID[o] for o in ("table_sum", "table_average", "table_max", "table_min")])
_SCALE = {CONSTANTS["const_1000"], CONSTANTS["const_1000000"]}
_COUNT = {CONSTANTS[f"const_{k}"] for k in (2, 3, 4, 5)}

PROGRAM_FEATURES: Final = (
    "n_steps",
    "op_first",
    "op_second",
    "op_last",
    "res_log_abs",
    "res_negative",
    "res_below_one",
    "res_integer",
    "res_is_bool",
    "res_vs_args",
    "n_numbers",
    "n_consts",
    "uses_100",
    "uses_scale",
    "uses_count",
    "uses_one",
    "uses_minus_one",
    "uses_table_op",
    "rank_min",
    "rank_max",
    "rank_mean",
    "label_overlap_mean",
    "label_overlap_min",
    "context_overlap_mean",
    "header_overlap_mean",
    "q_years_covered",
    "foreign_years",
    "same_row",
    "same_col",
    "n_facts",
    "uses_percent",
    "uses_variant",
    "uses_question_number",
    "row_label_overlap",
    "row_rank",
    "first_q_order",
    "anchored_share",
    *Q_FEATURES,
)
CATEGORICAL: Final = ("op_first", "op_second", "op_last")


def _nanagg(fn: Callable[..., np.ndarray], x: np.ndarray, axis: int) -> np.ndarray:
    """nanmin/nanmax/nanmean without the all-NaN-slice warning; such rows stay NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(x, axis=axis)


def program_features(
    ctx: QuestionContext, steps: np.ndarray, values: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    n, S, _ = steps.shape
    L = ctx.n_leaves
    out = np.full((n, len(PROGRAM_FEATURES)), np.nan, dtype=np.float32)
    rows = np.arange(n)
    last = lengths - 1
    ops = steps[:, :, 0]
    valid_step = np.arange(S)[None, :] < lengths[:, None]
    final = values[rows, last]
    is_bool = ops[rows, last] == OP_ID["greater"]

    out[:, 0] = lengths
    out[:, 1] = ops[:, 0]
    out[:, 2] = np.where(lengths > 1, ops[:, min(1, S - 1)], np.nan)
    out[:, 3] = ops[rows, last]
    out[:, 4] = np.where(is_bool, np.nan, log_abs(final))
    out[:, 5] = np.where(is_bool, np.nan, final < 0)
    out[:, 6] = np.where(is_bool, np.nan, np.abs(final) < 1)
    out[:, 7] = np.where(is_bool, np.nan, np.isclose(final, np.round(final)))
    out[:, 8] = is_bool

    # Leaf arguments of every arithmetic step, flattened to (n, 2S).
    is_table = np.isin(ops, _TABLE_OPS) & valid_step
    arith = valid_step & ~is_table
    args = steps[:, :, 1:3].reshape(n, 2 * S)
    arg_ok = np.repeat(arith, 2, axis=1) & (args >= 0) & (args < L)
    safe = np.where(arg_ok, args, 0)
    F = ctx.leaf_feats[safe]  # (n, 2S, N_ARG)
    F = np.where(arg_ok[:, :, None], F, np.nan)
    kind = F[:, :, A["kind"]]
    is_num = kind == 0
    vals = np.where(arg_ok, ctx.leaf_values[safe], np.nan)

    out[:, 9] = np.where(
        is_bool, np.nan, log_abs(final) - _nanagg(np.nanmax, log_abs(np.abs(vals)), 1)
    )
    out[:, 10] = is_num.sum(1)
    out[:, 11] = (kind == 1).sum(1)
    cv = np.where(kind == 1, vals, np.nan)
    out[:, 12] = (cv == 100).any(1)
    out[:, 13] = np.isin(cv, list(_SCALE)).any(1)
    out[:, 14] = np.isin(cv, list(_COUNT)).any(1)
    out[:, 15] = (cv == 1).any(1)
    out[:, 16] = (cv == -1).any(1)
    out[:, 17] = is_table.any(1)
    num_rank = np.where(is_num, F[:, :, A["fact_rank"]], np.nan)
    out[:, 18] = _nanagg(np.nanmin, num_rank, 1)
    out[:, 19] = _nanagg(np.nanmax, num_rank, 1)
    out[:, 20] = _nanagg(np.nanmean, num_rank, 1)
    lab = np.where(is_num, F[:, :, A["label_overlap"]], np.nan)
    out[:, 21] = _nanagg(np.nanmean, lab, 1)
    out[:, 22] = _nanagg(np.nanmin, lab, 1)
    out[:, 23] = _nanagg(np.nanmean, np.where(is_num, F[:, :, A["context_overlap"]], np.nan), 1)
    out[:, 24] = _nanagg(np.nanmean, np.where(is_num, F[:, :, A["header_overlap"]], np.nan), 1)

    years = np.where(arg_ok, ctx.leaf_year[safe], np.nan)
    if ctx.q_years:
        covered = np.zeros(n)
        for y in ctx.q_years:
            covered += (years == y).any(1)
        out[:, 25] = covered / len(ctx.q_years)
        foreign = np.isfinite(years) & ~np.isin(years, ctx.q_years)
        out[:, 26] = foreign.sum(1)

    row = np.where(arg_ok, ctx.leaf_row[safe], -1).astype(np.float64)
    row[row < 0] = np.nan
    out[:, 27] = _nanagg(np.nanmax, row, 1) == _nanagg(np.nanmin, row, 1)
    col = np.where(arg_ok, ctx.leaf_col[safe], -1).astype(np.float64)
    col[col < 0] = np.nan
    out[:, 28] = _nanagg(np.nanmax, col, 1) == _nanagg(np.nanmin, col, 1)
    fact = np.where(arg_ok & is_num, ctx.leaf_fact[safe], -99)
    srt = np.sort(fact, axis=1)
    distinct = (np.diff(srt, axis=1) != 0) & (srt[:, 1:] != -99)
    out[:, 29] = distinct.sum(1) + (srt[:, 0] != -99)
    out[:, 30] = (F[:, :, A["percent"]] == 1).any(1)
    out[:, 31] = (F[:, :, A["variant"]] > 0).any(1)
    out[:, 32] = (F[:, :, A["fact_rank"]] == -1).any(1)

    if len(ctx.rows):
        t_row = np.where(is_table, steps[:, :, 1], -1)
        t_any = (t_row >= 0).any(1)
        t_idx = np.where(t_any, t_row.max(1), 0)
        out[:, 33] = np.where(t_any, ctx.row_feats[t_idx, A["label_overlap"]], np.nan)
        out[:, 34] = np.where(t_any, ctx.row_feats[t_idx, A["fact_rank"]], np.nan)

    # Operand order of the first step: does its left argument come up first in the question?
    qpos = F[:, :, A["q_pos"]]
    out[:, 35] = np.sign(qpos[:, 0] - qpos[:, 1])
    anchored = np.isfinite(qpos) & is_num
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:, 36] = anchored.sum(1) / is_num.sum(1)

    out[:, 37:] = ctx.q_feats
    return out


def pack(
    programs: list[tuple[list[tuple[int, int, int]], list[float | bool]]], max_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad (actions, values) pairs into the arrays ``program_features`` takes."""
    n = len(programs)
    steps = np.zeros((n, max_steps, 3), dtype=np.int64)
    values = np.zeros((n, max_steps), dtype=np.float64)
    lengths = np.zeros(n, dtype=np.int64)
    for i, (acts, vals) in enumerate(programs):
        acts = [a for a in acts if a[0] != STOP]
        lengths[i] = len(acts)
        steps[i, : len(acts)] = acts
        values[i, : len(acts)] = [float(v) for v in vals[: len(acts)]]
    return steps, values, lengths


@dataclass
class ProgramRanker:
    """A LightGBM binary model over ``PROGRAM_FEATURES``; ``score`` returns P(correct)."""

    booster: lgb.Booster

    def score(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return np.asarray(self.booster.predict(X, num_threads=1), dtype=np.float64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> ProgramRanker:
        return cls(lgb.Booster(model_file=str(path)))
