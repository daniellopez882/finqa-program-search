from __future__ import annotations

from pathlib import Path

import pytest

from finqa_search.context import QuestionContext, build_context
from finqa_search.dataset import Example
from finqa_search.retrieval.retriever import FactRetriever, Ranking, RetrieverConfig, fit_idf
from finqa_search.space import ActionSpace


@pytest.fixture
def example() -> Example:
    return Example(
        id="ACME/2009/page_12.pdf-1",
        filename="ACME/2009/page_12.pdf",
        question="what was the percent change in net revenue from 2008 to 2009?",
        pre_text=[
            "acme corporation reports in millions of dollars .",
            "net revenue increased in 2009 due to higher volume .",
        ],
        post_text=["operating expenses were 1,200 in 2009 ."],
        table=[
            ["( in millions )", "2009", "2008", "2007"],
            ["net revenue", "$ 5,829", "$ 5,735", "$ 5,516"],
            ["operating income", "310", "295", "288"],
        ],
        program="subtract(5829, 5735), divide(#0, 5735)",
        exe_ans=0.01639,
        gold_inds={"table_1": "the 2009 of net revenue is $ 5,829 ;"},
    )


@pytest.fixture
def ranking(example: Example) -> Ranking:
    retriever = FactRetriever(fit_idf([example]), RetrieverConfig(use_cross_encoder=False))
    return retriever.rank(example)


@pytest.fixture
def ctx(example: Example, ranking: Ranking) -> QuestionContext:
    return build_context(example, ranking)


@pytest.fixture
def space(ctx: QuestionContext) -> ActionSpace:
    return ActionSpace(ctx, max_steps=4)


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The bundled 50-question run, done once per test session."""
    from finqa_search.demo import run_demo

    out = tmp_path_factory.mktemp("demo")
    run_demo(out, log=lambda _: None)
    return out
