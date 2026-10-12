"""The typed program space the searches walk.

A state is a partial program; every action appends one step or stops. Arguments are
indices: ``0 <= i < L`` is a leaf of the context (numbers, then constants), ``L + k`` is
the result of step ``k``, and a table operation's first index is a row of
``context.rows``. Typing rules:

* constants only fill the (operation, slot) pairs in ``CONSTANT_SLOTS``; two constants
  never meet in one step;
* ``exp`` needs a step result as its base;
* ``greater`` yields yes/no, so it must be the final step;
* a program may stop only when every earlier step feeds a later one, and an action is
  legal only if the steps left can still consume every dangling result;
* add and multiply are generated in one canonical argument order (results, then
  numbers, then constants), which is the commutativity half of the transposition table.

Feature rows for the leaf-only actions are built once per question and reused by every
state; only the few columns that describe the state are rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finqa_search.context import CONSTANT_SLOTS, QuestionContext
from finqa_search.dsl import (
    COMMUTATIVE,
    TABLE_OPS,
    Arg,
    Const,
    NoArg,
    Num,
    Program,
    Ref,
    Row,
    Step,
    apply_table_op,
    value_key,
)
from finqa_search.features import (
    ACTION_FEATURES,
    KIND_REF,
    N_ARG,
    OP_ID,
    OPS,
    PAIR_FEATURES,
    RESULT_FEATURES,
    STATE_FEATURES,
    STOP,
    A,
    log_abs,
    result_features,
    scalar_log_abs,
)

ADD, SUB, MUL, DIV, EXP, GT = (
    OP_ID[o] for o in ("add", "subtract", "multiply", "divide", "exp", "greater")
)
TABLE_OP_IDS = tuple(OP_ID[o] for o in TABLE_OPS)
ARITH_IDS = (ADD, SUB, MUL, DIV, EXP, GT)

N_FEAT = len(ACTION_FEATURES)
_S0 = 0
_A0 = len(STATE_FEATURES)
_B0 = _A0 + N_ARG
_P0 = _B0 + N_ARG
_R0 = _P0 + len(PAIR_FEATURES)
_QPOS = A["q_pos"]
_Q0 = _R0 + len(RESULT_FEATURES)

Action = tuple[int, int, int]
STOP_ACTION: Action = (STOP, -1, -1)


@dataclass(frozen=True, slots=True)
class State:
    steps: tuple[Action, ...] = ()
    values: tuple[float | bool, ...] = ()
    forms: tuple[str, ...] = ()
    consumed: frozenset[int] = frozenset()
    leaf_sets: tuple[frozenset[int], ...] = ()
    key: str = ""

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def open_steps(self) -> list[int]:
        return [i for i in range(len(self.steps)) if i not in self.consumed]

    @property
    def can_stop(self) -> bool:
        return bool(self.steps) and self.open_steps == [len(self.steps) - 1]

    @property
    def answer(self) -> float | bool | None:
        return self.values[-1] if self.values else None


@dataclass
class ActionBatch:
    """Legal actions of one state, their step results and their feature rows."""

    actions: np.ndarray  # (n, 3) int64: op, a, b
    results: np.ndarray  # (n,) float64; NaN for stop, 0/1 for greater
    features: np.ndarray  # (n, N_FEAT) float32

    def __len__(self) -> int:
        return len(self.actions)

    def action(self, i: int) -> Action:
        op, a, b = self.actions[i]
        return (int(op), int(a), int(b))


def safe_apply(op: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        if op == ADD:
            r = x + y
        elif op == SUB:
            r = x - y
        elif op == MUL:
            r = x * y
        elif op == DIV:
            r = np.where(y != 0, x / np.where(y == 0, 1.0, y), np.nan)
        elif op == EXP:
            r = np.where((x > 0) | (np.round(y) == y), np.power(np.abs(x), y), np.nan)
            r = np.where(x < 0, np.where(np.mod(y, 2) == 0, r, -r), r)
        else:  # greater
            r = (x > y).astype(np.float64)
    return r


class ActionSpace:
    def __init__(self, ctx: QuestionContext, max_steps: int = 4) -> None:
        self.ctx = ctx
        self.max_steps = max_steps
        self.L = ctx.n_leaves
        self.n_numbers = sum(1 for leaf in ctx.leaves if isinstance(leaf.arg, Num))
        self._const_name = [
            leaf.arg.name if isinstance(leaf.arg, Const) else "" for leaf in ctx.leaves
        ]
        self._build_static()

    # -- leaf-only actions, built once -----------------------------------------------------

    def _slot_ok(self, op_name: str, idx: int, slot: int) -> bool:
        name = self._const_name[idx] if 0 <= idx < self.L else ""
        return not name or (op_name, slot) in CONSTANT_SLOTS.get(name, frozenset())

    def typed_pairs(self, op: int, left: list[int], right: list[int]) -> list[tuple[int, int]]:
        """Typed argument pairs for one operation (leaf or ref indices)."""
        name = OPS[op]
        out: list[tuple[int, int]] = []
        for a in left:
            for b in right:
                if a == b:
                    continue
                if self._const_name_of(a) and self._const_name_of(b):
                    continue
                if op in (ADD, MUL) and not self._canonical(a, b):
                    continue
                if op == GT and (self._const_name_of(a) or self._const_name_of(b)):
                    continue
                if op == EXP and a < self.L:
                    continue
                if not (self._slot_ok(name, a, 0) and self._slot_ok(name, b, 1)):
                    continue
                out.append((a, b))
        return out

    def _const_name_of(self, idx: int) -> str:
        return self._const_name[idx] if 0 <= idx < self.L else ""

    def _canonical(self, a: int, b: int) -> bool:
        """Canonical order for commutative ops: step results, then numbers, then constants."""
        return self._order_key(a) < self._order_key(b)

    def _order_key(self, idx: int) -> tuple[int, int]:
        return (0, idx) if idx >= self.L else (1, idx)

    def _build_static(self) -> None:
        ctx = self.ctx
        leaves = list(range(self.L))
        acts: list[tuple[int, int, int]] = []
        for op in (ADD, SUB, MUL, DIV, GT):
            acts.extend((op, a, b) for a, b in self.typed_pairs(op, leaves, leaves))
        arr = np.array(acts, dtype=np.int64).reshape(-1, 3)
        va = ctx.leaf_values[arr[:, 1]] if len(arr) else np.zeros(0)
        vb = ctx.leaf_values[arr[:, 2]] if len(arr) else np.zeros(0)
        res = np.empty(len(arr))
        for op in (ADD, SUB, MUL, DIV, GT):
            m = arr[:, 0] == op
            res[m] = safe_apply(op, va[m], vb[m])
        keep = np.isfinite(res) & ~((arr[:, 0] == SUB) & (va == vb))
        keep &= ~((arr[:, 0] == DIV) & (va == vb))
        arr, va, vb, res = arr[keep], va[keep], vb[keep], res[keep]

        X = np.full((len(arr), N_FEAT), np.nan, dtype=np.float32)
        X[:, _S0] = arr[:, 0]
        X[:, _A0:_B0] = ctx.leaf_feats[arr[:, 1]]
        X[:, _B0:_P0] = ctx.leaf_feats[arr[:, 2]]
        ra, rb = ctx.leaf_row[arr[:, 1]], ctx.leaf_row[arr[:, 2]]
        ca, cb = ctx.leaf_col[arr[:, 1]], ctx.leaf_col[arr[:, 2]]
        fa, fb = ctx.leaf_fact[arr[:, 1]], ctx.leaf_fact[arr[:, 2]]
        ya, yb = ctx.leaf_year[arr[:, 1]], ctx.leaf_year[arr[:, 2]]
        X[:, _P0] = np.where((ra >= 0) & (rb >= 0), ra == rb, np.nan)
        X[:, _P0 + 1] = np.where((ca >= 0) & (cb >= 0), ca == cb, np.nan)
        X[:, _P0 + 2] = np.where((fa >= 0) & (fb >= 0), fa == fb, np.nan)
        X[:, _P0 + 3] = np.sign(ya - yb)
        X[:, _P0 + 4] = log_abs(va) - log_abs(vb)
        X[:, _P0 + 5] = va > vb
        X[:, _P0 + 6] = np.sign(X[:, _A0 + _QPOS] - X[:, _B0 + _QPOS])
        X[:, _R0:_Q0] = result_features(res)
        X[:, _Q0:] = ctx.q_feats

        # Table operations aggregate one retrieved row.
        t_acts: list[tuple[int, int, int]] = []
        t_res: list[float] = []
        for r, row in enumerate(ctx.rows):
            for op_name in TABLE_OPS:
                t_acts.append((OP_ID[op_name], r, -1))
                t_res.append(apply_table_op(op_name, list(row.values)))
        if t_acts:
            tarr = np.array(t_acts, dtype=np.int64)
            tres = np.array(t_res)
            TX = np.full((len(tarr), N_FEAT), np.nan, dtype=np.float32)
            TX[:, _S0] = tarr[:, 0]
            TX[:, _A0:_B0] = ctx.row_feats[tarr[:, 1]]
            TX[:, _R0:_Q0] = result_features(tres)
            TX[:, _Q0:] = ctx.q_feats
            arr = np.vstack([arr, tarr])
            res = np.concatenate([res, tres])
            X = np.vstack([X, TX])
        self.static_actions = arr
        self.static_results = res
        self.static_features = X
        self._static_index = {tuple(int(v) for v in a): i for i, a in enumerate(arr)}

    # -- states ----------------------------------------------------------------------------

    def root(self) -> State:
        return State()

    def _leaf_form(self, idx: int) -> str:
        return value_key(self.ctx.leaves[idx].value)

    def _arg_form(self, state_forms: tuple[str, ...], idx: int) -> str:
        return state_forms[idx - self.L] if idx >= self.L else self._leaf_form(idx)

    def child(self, state: State, action: Action) -> State:
        op, a, b = action
        if op == STOP:
            raise ValueError("stop has no child state")
        if op in TABLE_OP_IDS:
            row = self.ctx.rows[a]
            value: float | bool = apply_table_op(OPS[op], list(row.values))
            form = f"{OPS[op]}(row:{row.label},none)"
            used: set[int] = set()
            leaf_set: frozenset[int] = frozenset()
        else:
            x = self._value(state, a)
            y = self._value(state, b)
            r = float(safe_apply(op, np.array([x]), np.array([y]))[0])
            if not np.isfinite(r):
                raise ValueError(f"{OPS[op]}({x}, {y}) is undefined")
            value = bool(r) if op == GT else r
            parts = [self._arg_form(state.forms, a), self._arg_form(state.forms, b)]
            if OPS[op] in COMMUTATIVE:
                parts.sort()
            form = f"{OPS[op]}({parts[0]},{parts[1]})"
            used = {i - self.L for i in (a, b) if i >= self.L}
            leaf_set = frozenset(
                set().union(*(state.leaf_sets[i - self.L] if i >= self.L else {i} for i in (a, b)))
            )
        forms = (*state.forms, form)
        return State(
            steps=(*state.steps, action),
            values=(*state.values, value),
            forms=forms,
            consumed=state.consumed | used,
            leaf_sets=(*state.leaf_sets, leaf_set),
            key="|".join(sorted(forms)) + "=>" + form,
        )

    def _value(self, state: State, idx: int) -> float:
        if idx >= self.L:
            v = state.values[idx - self.L]
            if isinstance(v, bool):
                raise ValueError("a yes/no result cannot be an argument")
            return v
        return float(self.ctx.leaf_values[idx])

    def program(self, state: State) -> Program:
        return Program(tuple(self.step(state, act) for act in state.steps))

    def step(self, state: State, action: Action) -> Step:
        op, a, b = action
        if op in TABLE_OP_IDS:
            return Step(OPS[op], Row(self.ctx.rows[a].label), NoArg())
        return Step(OPS[op], self._arg(a), self._arg(b))

    def _arg(self, idx: int) -> Arg:
        if idx >= self.L:
            return Ref(idx - self.L)
        return self.ctx.leaves[idx].arg

    # -- legal actions ---------------------------------------------------------------------

    def _ref_feats(self, state: State) -> np.ndarray:
        d = state.depth
        F = np.full((d, N_ARG), np.nan)
        for k in range(d):
            v = state.values[k]
            fv = float(v)
            F[k, A["kind"]] = KIND_REF
            F[k, A["log_abs"]] = scalar_log_abs(fv)
            F[k, A["negative"]] = float(fv < 0)
            F[k, A["ref_op"]] = state.steps[k][0]
            F[k, A["ref_is_last"]] = float(k == d - 1)
            F[k, A["ref_size"]] = len(state.leaf_sets[k]) or 1
            ranks = [
                self.ctx.leaf_feats[i, A["fact_rank"]]
                for i in state.leaf_sets[k]
                if not np.isnan(self.ctx.leaf_feats[i, A["fact_rank"]])
            ]
            if ranks:
                F[k, A["fact_rank"]] = min(ranks)
        return F

    def legal(self, state: State) -> ActionBatch:
        d = state.depth
        open_now = state.open_steps
        remaining = self.max_steps - d
        parts_a: list[np.ndarray] = []
        parts_r: list[np.ndarray] = []
        parts_x: list[np.ndarray] = []
        last = state.values[-1] if state.values else None
        last_is_bool = isinstance(last, bool)

        if remaining > 0 and not last_is_bool:
            # Leaf-only steps leave every open result dangling and add one more.
            if len(open_now) + 1 - 1 <= remaining - 1:
                acts = self.static_actions
                mask = np.ones(len(acts), dtype=bool)
                if len(open_now) > 0:
                    mask &= acts[:, 0] != GT
                for prev in state.steps:
                    j = self._static_index.get(prev)
                    if j is not None:
                        mask[j] = False
                parts_a.append(acts[mask])
                parts_r.append(self.static_results[mask])
                parts_x.append(self.static_features[mask].copy())
            if d > 0:
                ra, rr, rx = self._ref_block(state, open_now, remaining)
                if len(ra):
                    parts_a.append(ra)
                    parts_r.append(rr)
                    parts_x.append(rx)

        if state.can_stop:
            sx = np.full((1, N_FEAT), np.nan, dtype=np.float32)
            sx[0, _S0] = STOP
            sx[0, _A0:_B0] = self._ref_feats(state)[-1]
            sx[0, _R0:_Q0] = result_features(np.array([float(last)]))[0]
            sx[0, _Q0:] = self.ctx.q_feats
            parts_a.append(np.array([STOP_ACTION], dtype=np.int64))
            parts_r.append(np.array([np.nan]))
            parts_x.append(sx)

        if not parts_a:
            return ActionBatch(
                np.zeros((0, 3), dtype=np.int64), np.zeros(0), np.zeros((0, N_FEAT), np.float32)
            )
        actions = np.vstack(parts_a)
        X = np.vstack(parts_x)
        X[:, 1] = d
        X[:, 2] = len(open_now)
        X[:, 3] = state.steps[-1][0] if d else np.nan
        X[:, 4] = scalar_log_abs(float(last)) if d else np.nan
        return ActionBatch(actions, np.concatenate(parts_r), X)

    def _ref_block(
        self, state: State, open_now: list[int], remaining: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        L = self.L
        refs = [L + k for k in range(state.depth) if not isinstance(state.values[k], bool)]
        leaves = list(range(L))
        open_set = set(open_now)
        acts: list[tuple[int, int, int]] = []
        for op in ARITH_IDS:
            pairs = self.typed_pairs(op, refs, leaves) + self.typed_pairs(op, leaves, refs)
            pairs += self.typed_pairs(op, refs, refs)
            for a, b in pairs:
                used = {i - L for i in (a, b) if i >= L} & open_set
                n_open_after = len(open_set - used) + 1
                if n_open_after - 1 > remaining - 1:
                    continue
                if op == GT and n_open_after != 1:
                    continue
                acts.append((op, a, b))
        if not acts:
            return np.zeros((0, 3), np.int64), np.zeros(0), np.zeros((0, N_FEAT), np.float32)
        arr = np.array(acts, dtype=np.int64)
        values = np.concatenate([self.ctx.leaf_values, [float(v) for v in state.values]])
        va, vb = values[arr[:, 1]], values[arr[:, 2]]
        res = np.empty(len(arr))
        for op in ARITH_IDS:
            m = arr[:, 0] == op
            if m.any():
                res[m] = safe_apply(op, va[m], vb[m])
        keep = np.isfinite(res) & ~((np.isin(arr[:, 0], (SUB, DIV, GT))) & (va == vb))
        arr, va, vb, res = arr[keep], va[keep], vb[keep], res[keep]
        argF = np.vstack([self.ctx.leaf_feats, self._ref_feats(state)])
        X = np.full((len(arr), N_FEAT), np.nan, dtype=np.float32)
        X[:, _S0] = arr[:, 0]
        X[:, _A0:_B0] = argF[arr[:, 1]]
        X[:, _B0:_P0] = argF[arr[:, 2]]
        X[:, _P0 + 4] = log_abs(va) - log_abs(vb)
        X[:, _P0 + 5] = va > vb
        X[:, _P0 + 6] = np.sign(X[:, _A0 + _QPOS] - X[:, _B0 + _QPOS])
        X[:, _R0:_Q0] = result_features(res)
        X[:, _Q0:] = self.ctx.q_feats
        return arr, res, X

    # -- gold programs ---------------------------------------------------------------------

    def encode(self, program: Program) -> list[Action] | None:
        """The action sequence that builds ``program`` here, or None if it is unreachable."""
        index = self.ctx.leaf_index()
        rows = {row.label: i for i, row in enumerate(self.ctx.rows)}
        out: list[Action] = []
        for step in program.steps:
            op = OP_ID[step.op]
            if step.op in TABLE_OPS:
                if not isinstance(step.a, Row) or step.a.label not in rows:
                    return None
                out.append((op, rows[step.a.label], -1))
                continue
            idx: list[int] = []
            for arg in (step.a, step.b):
                if isinstance(arg, Ref):
                    idx.append(self.L + arg.step)
                elif isinstance(arg, Num | Const):
                    j = index.get(value_key(arg.value))
                    if j is None:
                        return None
                    idx.append(j)
                else:
                    return None
            a, b = idx
            if op in (ADD, MUL) and not self._canonical(a, b):
                a, b = b, a
            out.append((op, a, b))
        return out
