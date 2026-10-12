from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from finqa_search.cli import app

runner = CliRunner()


def test_help_lists_the_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("fetch", "retrieve", "train", "solve", "run", "view", "demo"):
        assert command in result.output


def test_program_prints_steps_and_canonical_form() -> None:
    result = runner.invoke(app, ["program", "subtract(5829, 5735), divide(#0, 5735)"])
    assert result.exit_code == 0
    assert "#0 subtract(5829, 5735) = 94.0" in result.output
    assert "canonical: divide(subtract(5829.0,5735.0),5735.0)" in result.output


def test_program_reports_errors_with_exit_code_2() -> None:
    result = runner.invoke(app, ["program", "divide(5, 0)"])
    assert result.exit_code == 2


def test_program_runs_table_operations(tmp_path: Path) -> None:
    table = tmp_path / "t.json"
    table.write_text('[["", "2009", "2008"], ["revenue", "$ 10", "$ 14"]]', encoding="utf-8")
    result = runner.invoke(app, ["program", "table_average(revenue, none)", "--table", str(table)])
    assert result.exit_code == 0
    assert "= 12.0" in result.output


def test_view_writes_html_and_mermaid(demo_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "tree.html"
    result = runner.invoke(app, ["view", str(demo_dir / "trace.json"), "--out", str(out)])
    assert result.exit_code == 0
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
    sketch = runner.invoke(app, ["view", str(demo_dir / "trace.json"), "--mermaid"])
    assert sketch.output.startswith("flowchart LR")
