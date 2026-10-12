"""Training data and fitting for the action prior and the program ranker.

Prior: every state along a gold program is a ranking group whose positive is the gold
next step (or stop) and whose negatives are sampled legal actions.

Ranker: candidate programs come from searches with a prior that never saw the question
(3-fold cross-fitting on train), from ancestral samples of that prior and from uniform
draws of the depth-2 space; the label is whether the program executes to the gold answer.
A second round adds every program an MCTS run scores when it is steered by out-of-fold
models: those are the programs a search finds attractive, which the first round's
candidates under-represent.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np

from finqa_search.context import ContextConfig, QuestionContext, build_context
from finqa_search.dataset import Example
from finqa_search.dsl import answers_match
from finqa_search.features import ACTION_CATEGORICAL, ACTION_FEATURES, STOP
from finqa_search.prior import LightGBMPrior, Prior
from finqa_search.ranker import CATEGORICAL, PROGRAM_FEATURES, ProgramRanker, pack, program_features
from finqa_search.retrieval.retriever import Ranking
from finqa_search.search.baselines import beam_finished
from finqa_search.search.mcts import MCTS, MCTSConfig
from finqa_search.search.problem import ProgramProblem
from finqa_search.space import STOP_ACTION, Action, ActionBatch, ActionSpace, State


@dataclass(frozen=True)
class PriorConfig:
    negatives: int = 48
    objective: str = "lambdarank"
    num_leaves: int = 63
    # Every expansion runs the prior over a few thousand rows, so its size is search cost.
    rounds: int = 200
    learning_rate: float = 0.05
    min_data_in_leaf: int = 40
    feature_fraction: float = 0.8
    seed: int = 0


@dataclass(frozen=True)
class RankerConfig:
    folds: int = 3
    beam: int = 8
    samples: int = 16
    uniform: int = 48
    num_leaves: int = 63
    rounds: int = 300
    learning_rate: float = 0.05
    min_data_in_leaf: int = 40
    seed: int = 0
    search_simulations: int = 64  # second round: MCTS candidates; 0 skips it


@dataclass
class Item:
    """One question ready for search: the context, the space and the gold trajectory."""

    example: Example
    ctx: QuestionContext
    space: ActionSpace
    gold: list[Action] | None = field(default=None)


def find_action(batch: ActionBatch, action: Action) -> int | None:
    hit = np.flatnonzero((batch.actions == np.array(action)).all(axis=1))
    return int(hit[0]) if len(hit) else None


def gold_trajectory(space: ActionSpace, example: Example) -> list[Action] | None:
    """The gold program as legal actions in this space, ending with stop; None if unreachable."""
    program = example.gold_program()
    if program is None or len(program) > space.max_steps:
        return None
    acts = space.encode(program)
    if acts is None:
        return None
    state = space.root()
    for act in acts:
        if find_action(space.legal(state), act) is None:
            return None
        state = space.child(state, act)
    if not state.can_stop:
        return None
    return [*acts, STOP_ACTION]


def make_item(
    example: Example, ranking: Ranking, ctx_config: ContextConfig, max_steps: int
) -> Item:
    ctx = build_context(example, ranking, ctx_config)
    space = ActionSpace(ctx, max_steps=max_steps)
    return Item(example, ctx, space, gold_trajectory(space, example))


# -- prior -----------------------------------------------------------------------------


def prior_rows(
    item: Item, rng: np.random.Generator, negatives: int
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    assert item.gold is not None
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    groups: list[int] = []
    state: State = item.space.root()
    for act in item.gold:
        batch = item.space.legal(state)
        pos = find_action(batch, act)
        if pos is None:
            break
        others = np.delete(np.arange(len(batch)), pos)
        if len(others) > negatives:
            others = rng.choice(others, size=negatives, replace=False)
        idx = np.concatenate([[pos], others])
        xs.append(batch.features[idx])
        y = np.zeros(len(idx))
        y[0] = 1.0
        ys.append(y)
        groups.append(len(idx))
        if act[0] == STOP:
            break
        state = item.space.child(state, act)
    if not xs:
        return np.zeros((0, len(ACTION_FEATURES)), np.float32), np.zeros(0), []
    return np.vstack(xs), np.concatenate(ys), groups


def fit_prior(items: Sequence[Item], config: PriorConfig, threads: int = 6) -> LightGBMPrior:
    rng = np.random.default_rng(config.seed)
    rows = [(*prior_rows(item, rng, config.negatives), True) for item in items if item.gold]
    return fit_prior_rows(rows, config, threads)


def fit_prior_rows(
    rows: Sequence[tuple[np.ndarray, np.ndarray, list[int], bool]],
    config: PriorConfig,
    threads: int = 6,
) -> LightGBMPrior:
    """Fit on per-question (features, labels, group sizes, reachable) tuples."""
    used = [r for r in rows if r[3] and len(r[1])]
    X = np.vstack([r[0] for r in used])
    y = np.concatenate([r[1] for r in used])
    groups = [g for r in used for g in r[2]]
    params: dict[str, Any] = {
        "objective": config.objective,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_data_in_leaf,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": config.seed,
        "deterministic": True,
        "num_threads": threads,
        "verbose": -1,
    }
    if config.objective == "lambdarank":
        params["lambdarank_truncation_level"] = 20
    data = lgb.Dataset(
        X,
        label=y,
        group=groups if config.objective == "lambdarank" else None,
        feature_name=list(ACTION_FEATURES),
        categorical_feature=list(ACTION_CATEGORICAL),
        free_raw_data=True,
    )
    booster = lgb.train(params, data, num_boost_round=config.rounds)
    return LightGBMPrior(booster)


# -- ranker ----------------------------------------------------------------------------


def label_programs(
    item: Item, programs: list[tuple[list[Action], list[float | bool]]]
) -> np.ndarray:
    return np.array([answers_match(vals[-1], item.example.exe_ans) for _, vals in programs])


def fit_ranker(
    X: np.ndarray, y: np.ndarray, config: RankerConfig, threads: int = 6
) -> ProgramRanker:
    params: dict[str, Any] = {
        "objective": "binary",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_data_in_leaf,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": config.seed,
        "deterministic": True,
        "num_threads": threads,
        "verbose": -1,
    }
    data = lgb.Dataset(
        X,
        label=y.astype(np.float64),
        feature_name=list(PROGRAM_FEATURES),
        categorical_feature=list(CATEGORICAL),
    )
    return ProgramRanker(lgb.train(params, data, num_boost_round=config.rounds))


def ranker_matrix(
    item: Item, programs: list[tuple[list[Action], list[float | bool]]]
) -> np.ndarray:
    steps, values, lengths = pack(programs, item.space.max_steps)
    return program_features(item.ctx, steps, values, lengths)


def parallel_map(
    fn: Callable[[Any], Any],
    jobs: Sequence[Any],
    workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
) -> list[Any]:
    """``map`` over spawned processes (Windows-safe), or in-process for one worker."""
    if workers <= 1:
        if initializer:
            initializer(*initargs)
        return [fn(j) for j in jobs]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=initializer, initargs=initargs) as pool:
        return pool.map(fn, jobs, chunksize=max(1, len(jobs) // (workers * 8)))


# -- ranker candidates -----------------------------------------------------------------


def sample_programs(
    problem: ProgramProblem, n: int, rng: np.random.Generator, temperature: float = 1.0
) -> list[State]:
    """Ancestral samples from the prior (the problem memoises expansions per state)."""
    out: list[State] = []
    for _ in range(n):
        state = problem.root()
        while True:
            exp = problem.expand([state])[0]
            if not len(exp):
                break
            p = exp.priors ** (1.0 / temperature)
            act = exp.actions[int(rng.choice(len(exp), p=p / p.sum()))]
            if act == STOP_ACTION:
                out.append(state)
                break
            state = problem.child(state, act)
    return out


def uniform_programs(space: ActionSpace, n: int, rng: np.random.Generator) -> list[State]:
    """Uniform draws from the depth-2 space the exhaustive baseline enumerates."""
    out: list[State] = []
    first = space.static_actions
    if not len(first):
        return out
    L = space.L
    for _ in range(n):
        a = tuple(int(v) for v in first[rng.integers(len(first))])
        s1 = space.child(space.root(), a)  # type: ignore[arg-type]
        if rng.random() < 0.5 or isinstance(s1.answer, bool):
            out.append(s1)
            continue
        batch = space.legal(s1)
        uses = np.flatnonzero((batch.actions[:, 1] == L) | (batch.actions[:, 2] == L))
        if not len(uses):
            out.append(s1)
            continue
        s2 = space.child(s1, batch.action(int(rng.choice(uses))))
        out.append(s2 if s2.can_stop else s1)
    return out


def ranker_candidates(
    item: Item, prior: Prior, config: RankerConfig, rng: np.random.Generator
) -> list[State]:
    problem = ProgramProblem(item.space, prior, None, lookahead=0)
    states: dict[str, State] = {}
    for _, s in beam_finished(problem, config.beam):
        states[s.key] = s
    for s in sample_programs(problem, config.samples, rng):
        states[s.key] = s
    for s in uniform_programs(item.space, config.uniform, rng):
        states[s.key] = s
    if item.gold is not None:
        s = item.space.root()
        for act in item.gold[:-1]:
            s = item.space.child(s, act)
        states[s.key] = s
    return list(states.values())


def ranker_rows(
    item: Item, prior: Prior, config: RankerConfig, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = [s for s in ranker_candidates(item, prior, config, rng) if s.can_stop]
    if not states:
        return np.zeros((0, len(PROGRAM_FEATURES)), np.float32), np.zeros(0)
    progs = [(list(s.steps), list(s.values)) for s in states]
    X = ranker_matrix(item, progs)
    y = label_programs(item, progs).astype(np.float64)
    return X, y


def search_rows(
    item: Item, prior: Prior, ranker: ProgramRanker, simulations: int
) -> tuple[np.ndarray, np.ndarray]:
    """Every finished program one MCTS run scored, labelled by execution."""
    problem = ProgramProblem(item.space, prior, ranker)
    MCTS(problem, MCTSConfig(simulations=simulations)).run()
    states = [s for s, _ in problem.scored.values()]
    if not states:
        return np.zeros((0, len(PROGRAM_FEATURES)), np.float32), np.zeros(0)
    progs = [(list(s.steps), list(s.values)) for s in states]
    return ranker_matrix(item, progs), label_programs(item, progs).astype(np.float64)
