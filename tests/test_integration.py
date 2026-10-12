"""Checks against the real FinQA splits and the cross-encoder; skipped when they are absent."""

from __future__ import annotations

import numpy as np
import pytest

from finqa_search.dataset import SPLITS, leakage, load_split, report_of
from finqa_search.fetch import finqa_ready, model_ready
from finqa_search.gold import check_split, load_allowlist

pytestmark = pytest.mark.integration

needs_data = pytest.mark.skipif(not finqa_ready(), reason="FinQA not downloaded")
needs_model = pytest.mark.skipif(not model_ready(), reason="cross-encoder not downloaded")


@needs_data
def test_every_train_gold_program_reproduces_its_answer_or_is_allowlisted() -> None:
    report = check_split(load_split("train"), load_allowlist())
    assert report.unexpected == {}, report.unexpected
    assert report.stale == []
    assert set(report.allowed) == set(load_allowlist())


@needs_data
@pytest.mark.parametrize("split", ["dev", "test"])
def test_dev_and_test_gold_programs_all_reproduce(split: str) -> None:
    report = check_split(load_split(split), {})  # type: ignore[arg-type]
    assert report.unexpected == {}


@needs_data
def test_splits_share_no_question_and_no_report_page() -> None:
    splits = {name: load_split(name) for name in SPLITS}
    counts = leakage(splits)
    assert all(v == 0 for v in counts.values()), counts
    for examples in splits.values():
        assert len({e.id for e in examples}) == len(examples)


@needs_data
def test_split_sizes_match_the_release() -> None:
    sizes = {name: len(load_split(name)) for name in SPLITS}
    assert sizes == {"train": 6251, "dev": 883, "test": 1147}
    assert sum(sizes.values()) == 8281
    pages = {report_of(e.id) for e in load_split("test")}
    assert len(pages) < sizes["test"]


@needs_model
def test_cross_encoder_prefers_the_relevant_passage() -> None:
    from finqa_search.retrieval.cross_encoder import CrossEncoder

    ce = CrossEncoder()
    scores = ce.score(
        "what was the net revenue in 2009?",
        [
            "the 2009 of net revenue is $ 5,829 ;",
            "the board of directors met four times during the year .",
        ],
    )
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert scores[0] > scores[1]
