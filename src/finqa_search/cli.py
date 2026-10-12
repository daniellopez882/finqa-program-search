"""``finqa-search``: fetch data, rank facts, train the models, search and inspect traces."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from finqa_search import fetch as fetch_mod
from finqa_search.context import ContextConfig
from finqa_search.dataset import SPLITS, Split, load_split
from finqa_search.dsl import ProgramError, canonical, execute_steps, parse_program
from finqa_search.layout import idf_path, models_dir, rankings_path
from finqa_search.search.mcts import MCTSConfig
from finqa_search.solver import Method, MethodName

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Tree search over executable FinQA programs.",
)

Workers = Annotated[int, typer.Option(min=1, help="Worker processes.")]
SplitOpt = Annotated[str, typer.Option(help="train, dev or test.")]


def _split(name: str) -> Split:
    if name not in SPLITS:
        raise typer.BadParameter(f"split must be one of {', '.join(SPLITS)}")
    return name  # type: ignore[return-value]


def _method(
    name: str,
    width: int,
    simulations: int,
    trees: int,
    c_puct: float,
    fpu_reduction: float,
    prior_weight: float,
    seed: int,
) -> Method:
    valid: tuple[MethodName, ...] = ("greedy", "beam", "beam+ranker", "depth2+ranker", "mcts")
    if name not in valid:
        raise typer.BadParameter(f"method must be one of {', '.join(valid)}")
    return Method(
        name,  # type: ignore[arg-type]
        width=width,
        mcts=MCTSConfig(
            simulations=simulations, c_puct=c_puct, fpu_reduction=fpu_reduction, seed=seed
        ),
        trees=trees,
        prior_weight=prior_weight,
    )


FpuOpt = Annotated[float, typer.Option(help="First-play urgency reduction for MCTS.")]
WeightOpt = Annotated[float, typer.Option(help="Weight of log P_prior when picking the answer.")]


@app.command()
def fetch(
    data: Annotated[bool, typer.Option(help="Download the FinQA splits.")] = True,
    model: Annotated[bool, typer.Option(help="Download the cross-encoder ONNX.")] = True,
) -> None:
    """Download FinQA and the cross-encoder, checking every file against its pinned sha256."""
    if data:
        fetch_mod.fetch_finqa()
    if model:
        fetch_mod.fetch_model()


@app.command()
def verify() -> None:
    """Re-hash the downloaded files without touching the network."""
    if not fetch_mod.verify_all():
        raise typer.Exit(1)


@app.command()
def retrieve(
    split: Annotated[list[str] | None, typer.Option(help="Splits to rank (default: all).")] = None,
    workers: Workers = 6,
    bm25_only: Annotated[bool, typer.Option(help="Skip the cross-encoder.")] = False,
) -> None:
    """Fit BM25 IDF on train and cache the fact ranking of every question."""
    from finqa_search.evaluate import progress_printer
    from finqa_search.retrieval.batch import rank_examples
    from finqa_search.retrieval.retriever import (
        RetrieverConfig,
        fit_idf,
        save_idf,
        save_rankings,
    )

    idf = fit_idf(load_split("train"))
    save_idf(idf_path(), idf)
    config = RetrieverConfig(use_cross_encoder=not bm25_only)
    for name in split or list(SPLITS):
        examples = load_split(_split(name))
        start = time.perf_counter()
        ranked = rank_examples(examples, idf, config, workers, progress_printer(name, 500))
        save_rankings(rankings_path(name), ranked)
        typer.echo(f"{name}: {len(ranked)} rankings in {time.perf_counter() - start:.0f} s")


@app.command()
def train(
    workers: Workers = 6,
    out: Annotated[Path | None, typer.Option(help="Model directory.")] = None,
) -> None:
    """Fit the action prior and the program ranker on the training split."""
    from finqa_search.pipeline import TrainSetup
    from finqa_search.pipeline import train as fit
    from finqa_search.retrieval.retriever import load_rankings
    from finqa_search.training import PriorConfig, RankerConfig

    setup = TrainSetup(ContextConfig(), 4, PriorConfig(), RankerConfig())
    manifest = fit(
        load_split("train"),
        load_rankings(rankings_path("train")),
        setup,
        out or models_dir(),
        workers=workers,
        log=typer.echo,
    )
    typer.echo(json.dumps(manifest["seconds"], indent=1))


def _searcher(models: Path | None, uniform_prior: bool, temperature: float) -> Any:
    from finqa_search.evaluate import SearchSetup

    root = models or models_dir()
    prior_path = None if uniform_prior else root / "prior.txt"
    return SearchSetup(prior_path, root / "ranker.txt", temperature=temperature)


@app.command()
def solve(
    question_id: Annotated[str, typer.Argument(help="FinQA id, e.g. ADI/2009/page_49.pdf-1")],
    split: SplitOpt = "test",
    method: Annotated[str, typer.Option(help="greedy, beam, beam+ranker, depth2+ranker, mcts")] = (
        "mcts"
    ),
    simulations: int = 256,
    width: int = 5,
    trees: Annotated[int, typer.Option(min=1, help="Root-parallel trees.")] = 1,
    c_puct: float = 1.5,
    fpu_reduction: FpuOpt = 0.0,
    prior_weight: WeightOpt = 1.0,
    seed: int = 0,
    temperature: float = 1.0,
    models: Annotated[Path | None, typer.Option(help="Model directory.")] = None,
    uniform_prior: bool = False,
    llm_url: Annotated[
        str | None, typer.Option(help="OpenAI-compatible base URL to mix in an LLM prior.")
    ] = None,
    llm_model: Annotated[str | None, typer.Option(help="Model name for --llm-url.")] = None,
    llm_key_env: Annotated[str, typer.Option(help="Env var holding the API key.")] = (
        "OPENAI_API_KEY"
    ),
    out: Annotated[Path | None, typer.Option(help="Write the search-trace JSON here.")] = None,
    html: Annotated[Path | None, typer.Option(help="Write the HTML tree viewer here.")] = None,
) -> None:
    """Search one question and print (or write) its trace."""
    import os

    from finqa_search.retrieval.retriever import load_rankings
    from finqa_search.solver import solve as run
    from finqa_search.solver import trace_record
    from finqa_search.training import make_item
    from finqa_search.viewer import write_html

    examples = {e.id: e for e in load_split(_split(split))}
    if question_id not in examples:
        raise typer.BadParameter(f"{question_id} is not in the {split} split")
    ranking = load_rankings(rankings_path(split))[question_id]
    setup = _searcher(models, uniform_prior, temperature)
    prior, ranker = setup.load()
    if llm_url:
        from finqa_search.llm import LLMPrior, OpenAICompatibleProvider

        if not llm_model:
            raise typer.BadParameter("--llm-model is required with --llm-url")
        provider = OpenAICompatibleProvider(llm_url, llm_model, api_key=os.environ.get(llm_key_env))
        prior = LLMPrior(provider, prior)
    item = make_item(examples[question_id], ranking, setup.context, setup.max_steps)
    m = _method(method, width, simulations, trees, c_puct, fpu_reduction, prior_weight, seed)
    record = trace_record(item, run(item, m, prior, ranker, with_trace=m.name == "mcts"))
    text = json.dumps(record, indent=1)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        typer.echo(f"trace written to {out}")
    else:
        typer.echo(text)
    if html:
        if record.get("trace") is None:
            raise typer.BadParameter("--html needs --method mcts")
        typer.echo(f"viewer written to {write_html(record, html)}")


@app.command()
def run(
    split: SplitOpt = "dev",
    method: str = "mcts",
    simulations: int = 64,
    width: int = 5,
    trees: int = 1,
    c_puct: float = 1.5,
    fpu_reduction: FpuOpt = 0.0,
    prior_weight: WeightOpt = 1.0,
    limit: Annotated[int, typer.Option(help="First N questions (0 = all).")] = 0,
    workers: Workers = 6,
    models: Annotated[Path | None, typer.Option(help="Model directory.")] = None,
    out: Annotated[Path | None, typer.Option(help="Write one JSON record per question.")] = None,
) -> None:
    """Search every question of a split and print execution and program accuracy."""
    from finqa_search.evaluate import run_split, summarise
    from finqa_search.retrieval.retriever import load_rankings

    examples = load_split(_split(split))
    if limit:
        examples = examples[:limit]
    m = _method(method, width, simulations, trees, c_puct, fpu_reduction, prior_weight, 0)
    setup = _searcher(models, False, 1.0)
    records = run_split(examples, load_rankings(rankings_path(split)), [m], setup, workers)[m.label]
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
    typer.echo(json.dumps({m.label: summarise(records)}, indent=1))


@app.command()
def view(
    trace: Annotated[Path, typer.Argument(help="Trace JSON written by solve or demo.")],
    out: Annotated[Path, typer.Option(help="HTML file to write.")] = Path("tree.html"),
    mermaid: Annotated[bool, typer.Option(help="Print a Mermaid sketch instead.")] = False,
) -> None:
    """Render a search trace as a static HTML tree (visits, priors, values)."""
    from finqa_search.viewer import mermaid as to_mermaid
    from finqa_search.viewer import write_html

    record = json.loads(trace.read_text(encoding="utf-8"))
    if record.get("trace") is None:
        raise typer.BadParameter("this record has no search tree (only MCTS writes one)")
    if mermaid:
        typer.echo(to_mermaid(record))
        return
    typer.echo(f"viewer written to {write_html(record, out)}")


@app.command()
def program(
    text: Annotated[str, typer.Argument(help='e.g. "subtract(5829, 5735), divide(#0, 5735)"')],
    table: Annotated[Path | None, typer.Option(help="JSON table for table operations.")] = None,
) -> None:
    """Parse and execute one program; print each step's value and the canonical form."""
    rows = json.loads(table.read_text(encoding="utf-8")) if table else []
    try:
        parsed = parse_program(text)
        values = execute_steps(parsed, rows)
    except ProgramError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    for i, (step, value) in enumerate(zip(parsed.steps, values, strict=True)):
        typer.echo(f"#{i} {step.render()} = {value}")
    typer.echo(f"canonical: {canonical(parsed)}")


@app.command("check-gold")
def check_gold(
    split: SplitOpt = "train",
    allowlist: Annotated[Path | None, typer.Option(help="Override the shipped allowlist.")] = None,
) -> None:
    """Execute every gold program and compare with its answer and the noise allowlist."""
    from finqa_search.gold import check_split, load_allowlist

    allowed = load_allowlist(allowlist)
    report = check_split(load_split(_split(split)), allowed)
    typer.echo(json.dumps(report.summary(), indent=1))
    if report.unexpected:
        for ex_id, reason in report.unexpected.items():
            typer.echo(f"unexpected: {ex_id}: {reason}", err=True)
        raise typer.Exit(1)


@app.command()
def demo(
    out: Annotated[Path, typer.Option(help="Directory for models and the trace.")] = Path(
        "finqa-demo"
    ),
) -> None:
    """Offline run on the 50 bundled questions: BM25, tiny models, three searches, a trace."""
    from finqa_search.demo import run_demo

    run_demo(out, log=typer.echo)


if __name__ == "__main__":
    app(sys.argv[1:])
