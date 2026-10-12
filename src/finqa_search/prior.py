"""Action priors: a distribution over the legal next steps of a partial program.

``LightGBMPrior`` scores every legal action with one gradient-boosted model and turns
the scores into probabilities with a temperature softmax. ``UniformPrior`` is the
no-knowledge reference (PUCT with it is plain UCT over the typed space). An LLM can be
mixed in through ``finqa_search.llm.LLMPrior``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import lightgbm as lgb
import numpy as np

from finqa_search.space import ActionBatch, ActionSpace, State


class Prior(Protocol):
    def __call__(
        self, space: ActionSpace, states: list[State], batches: list[ActionBatch]
    ) -> list[np.ndarray]:
        """One probability vector per batch (same order as its actions)."""
        ...


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    if len(scores) == 0:
        return scores
    z = scores / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class UniformPrior:
    def __call__(
        self, space: ActionSpace, states: list[State], batches: list[ActionBatch]
    ) -> list[np.ndarray]:
        return [np.full(len(b), 1.0 / max(len(b), 1)) for b in batches]


@dataclass
class LightGBMPrior:
    booster: lgb.Booster
    temperature: float = 1.0
    calls: int = 0
    rows: int = 0

    def raw_scores(self, batches: list[ActionBatch]) -> list[np.ndarray]:
        sizes = [len(b) for b in batches]
        if not sum(sizes):
            return [np.zeros(0) for _ in batches]
        X = np.vstack([b.features for b in batches if len(b)])
        raw = np.asarray(self.booster.predict(X, raw_score=True, num_threads=1), dtype=np.float64)
        self.calls += 1
        self.rows += len(X)
        return np.split(raw, np.cumsum(sizes)[:-1])

    def __call__(
        self, space: ActionSpace, states: list[State], batches: list[ActionBatch]
    ) -> list[np.ndarray]:
        return [softmax(s, self.temperature) for s in self.raw_scores(batches)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path: Path, temperature: float = 1.0) -> LightGBMPrior:
        return cls(lgb.Booster(model_file=str(path)), temperature)
