"""Print the README's result tables and the accuracy curve from bench/results/*.json.

    uv run python bench/report.py            # the full run
    uv run python bench/report.py --smoke    # the smoke run

Every figure is formatted straight from the JSON, so the README can be regenerated
after a new benchmark run instead of being edited by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent / "results"


def load(prefix: str, stage: str) -> dict[str, Any] | None:
    path = RESULTS / f"{prefix}{stage}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def secs(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def measured_on(doc: dict[str, Any]) -> str:
    m = doc["machine"]
    return (
        f"Measured on {m['cpu']} ({m['physical_cores']} cores, {round(m['ram_gb'])} GB RAM, "
        f"{m['os']}, Python {m['python']}, CPU only) on {doc['date']}, commit "
        f"`{doc['git_sha'][:10]}`."
    )


def curve(test: dict[str, Any]) -> str:
    methods = test["metrics"]["methods"]
    sims = [int(k.split("-")[1]) for k in methods if k.startswith("mcts-") and " " not in k]
    sims.sort()
    acc = [100 * methods[f"mcts-{n}"]["execution_accuracy"] for n in sims]
    cov = [100 * methods[f"mcts-{n}"]["candidate_coverage"] for n in sims]
    greedy = 100 * methods["greedy"]["execution_accuracy"]
    beams = [k for k in methods if k.startswith("beam+ranker-")]
    best_beam = max(beams, key=lambda k: methods[k]["execution_accuracy"]) if beams else None
    lines = [
        "```mermaid",
        "xychart-beta",
        '    title "FinQA test: MCTS accuracy and candidate coverage by simulations"',
        "    x-axis [" + ", ".join(str(n) for n in sims) + "]",
        '    y-axis "% of test questions" 0 --> 100',
        "    line [" + ", ".join(f"{v:.1f}" for v in cov) + "]",
        "    line [" + ", ".join(f"{v:.1f}" for v in acc) + "]",
        "    line [" + ", ".join(f"{greedy:.1f}" for _ in sims) + "]",
    ]
    names = [
        "MCTS: a correct program was scored",
        "MCTS: execution accuracy",
        "greedy: execution accuracy (flat)",
    ]
    if best_beam:
        b = 100 * methods[best_beam]["execution_accuracy"]
        lines.append("    line [" + ", ".join(f"{b:.1f}" for _ in sims) + "]")
        names.append(f"{best_beam}: execution accuracy (flat)")
    lines.append("```")
    # Mermaid's xychart has no legend, so the line order is spelled out under it.
    lines += ["", "Lines, in drawing order: " + "; ".join(names) + "."]
    return "\n".join(lines)


def method_table(test: dict[str, Any]) -> str:
    rows = [
        "| method | execution acc. | program acc. | a correct program was scored | "
        "p50 s | p95 s | programs scored (mean) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, s in test["metrics"]["methods"].items():
        scored = s.get("programs_scored_mean")
        rows.append(
            f"| {label} | {pct(s['execution_accuracy'])} | {pct(s['program_accuracy'])} | "
            f"{pct(s['candidate_coverage'])} | {secs(s['seconds_p50'])} | "
            f"{secs(s['seconds_p95'])} | {'-' if not scored else f'{scored:,.0f}'} |"
        )
    return "\n".join(rows)


def retrieval_table(doc: dict[str, Any]) -> str:
    rows = ["| split | ranker | Recall@1 | Recall@3 | Recall@5 | all gold facts in top 3 |"]
    rows.append("| --- | --- | ---: | ---: | ---: | ---: |")
    names = {"bm25": "BM25", "hybrid": "BM25 + cross-encoder"}
    rows.extend(
        f"| {split} | {names[name]} | "
        f"{pct(r[f'{name}_recall@1'])} | {pct(r[f'{name}_recall@3'])} | "
        f"{pct(r[f'{name}_recall@5'])} | {pct(r[f'{name}_all_gold@3'])} |"
        for split, r in doc["metrics"]["recall"].items()
        for name in names
    )
    return "\n".join(rows)


def latency_table(doc: dict[str, Any]) -> str:
    rows = [
        "| workers | trees x simulations | p50 s | p95 s | execution acc. |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for label, r in doc["metrics"].items():
        per = r["simulations_per_tree"]
        shape = f"{r['trees']} x {per[0]}" if len(set(per)) == 1 else f"{r['trees']} x {per}"
        rows.append(
            f"| {label} | {shape} | {secs(r['seconds_p50'])} | {secs(r['seconds_p95'])} | "
            f"{pct(r['execution_accuracy'])} |"
        )
    return "\n".join(rows)


def tune_table(doc: dict[str, Any]) -> str:
    rows = ["| dev setting | execution acc. | p50 s |", "| --- | ---: | ---: |"]
    rows += [
        f"| {r['label']} | {pct(r['execution_accuracy'])} | {secs(r['seconds_p50'])} |"
        for r in doc["metrics"]["grid"]
    ]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    prefix = "smoke-" if parser.parse_args().smoke else ""
    sections: list[tuple[str, str]] = []
    if test := load(prefix, "test"):
        sections += [("curve", curve(test)), ("methods", method_table(test))]
        sections.append(("measured", measured_on(test)))
    if doc := load(prefix, "retrieval"):
        sections.append(("retrieval", retrieval_table(doc)))
    if doc := load(prefix, "latency"):
        sections.append(("latency", latency_table(doc)))
    if doc := load(prefix, "tune"):
        sections.append(("tune", tune_table(doc)))
    if doc := load(prefix, "trace"):
        sections.append(("trace", "```mermaid\n" + doc["metrics"]["mermaid"] + "\n```"))
    for name, text in sections:
        print(f"<!-- {name} -->\n{text}\n")


if __name__ == "__main__":
    main()
