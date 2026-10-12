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


def os_name(platform_string: str) -> str:
    """``Windows-10-10.0.19045-SP0`` -> ``Windows 10``; other systems are kept as they are."""
    parts = platform_string.split("-")
    return f"{parts[0]} {parts[1]}" if parts[0] == "Windows" and len(parts) > 1 else parts[0]


def measured_on(doc: dict[str, Any]) -> str:
    m = doc["machine"]
    return (
        f"Measured on {m['cpu']} ({m['physical_cores']} cores, {round(m['ram_gb'])} GB RAM, "
        f"{os_name(m['os'])}, Python {m['python']}, CPU only) on {doc['date']}, commit "
        f"`{doc['git_sha'][:7]}`."
    )


def questions(doc: dict[str, Any]) -> str:
    """Which questions a stage ran on, from its ``config``."""
    config = doc["config"]
    sample = config.get("sample")
    if not sample:
        return f"all {config['questions']:,} {config['split']} questions"
    return (
        f"a seeded random sample of {sample['questions']:,} of the {sample['of']:,} "
        f"{config['split']} questions (seed {sample['seed']})"
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


def reach_line(test: dict[str, Any]) -> str:
    first = next(iter(test["metrics"]["methods"].values()))
    return (
        f"The gold program is reachable in the search space for {pct(first['gold_reachable'])} "
        f"of these questions; that is the ceiling of every row."
    )


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


def retrieval_timing(doc: dict[str, Any]) -> str:
    workers = doc["config"]["workers"]
    rows = [
        f"| split | questions | wall s ({workers} workers) | p50 s per question | "
        "p95 s per question |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    recall = doc["metrics"]["recall"]
    for split, t in doc["metrics"]["timing"].items():
        if t.get("reused_cache"):
            continue
        n = recall.get(split, {}).get("questions_with_gold_facts")
        rows.append(
            f"| {split} | {'-' if n is None else f'{n:,}'} | {t['wall_seconds']:.0f} | "
            f"{secs(t['per_question_p50'])} | {secs(t['per_question_p95'])} |"
        )
    return "\n".join(rows)


def compare_table(doc: dict[str, Any]) -> str:
    base = doc["config"]["baseline"]
    rows = [
        f"| method | execution acc. | 95% interval | right where {base} is wrong | "
        f"wrong where {base} is right | sign test p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, r in doc["metrics"]["vs_greedy"].items():
        if label == base:
            continue
        lo, hi = r["wilson95"]
        rows.append(
            f"| {label} | {pct(r['execution_accuracy'])} | {pct(lo)} to {pct(hi)} | "
            f"{r[f'right_where_{base}_wrong']} | {r[f'wrong_where_{base}_right']} | "
            f"{r['sign_test_p']:.2g} |"
        )
    return "\n".join(rows)


def ladder_table(doc: dict[str, Any]) -> str:
    rows = [
        "| simulations | chosen program changed | newly right | newly wrong |",
        "| --- | ---: | ---: | ---: |",
    ]
    rows += [
        f"| {step['from'].split('-')[1]} to {step['to'].split('-')[1]} | "
        f"{step['programs_changed']} | {step['newly_right']} | {step['newly_wrong']} |"
        for step in doc["metrics"]["mcts_ladder"]
    ]
    return "\n".join(rows)


def latency_table(doc: dict[str, Any]) -> str:
    rows = [
        "| workers | trees x simulations | p50 s | p95 s | mean s | execution acc. |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label, r in doc["metrics"].items():
        per = r["simulations_per_tree"]
        shape = (
            f"{r['trees']} x {per[0]}"
            if len(set(per)) == 1
            else f"{r['trees']} x {min(per)}-{max(per)}"
        )
        rows.append(
            f"| {label} | {shape} | {secs(r['seconds_p50'])} | {secs(r['seconds_p95'])} | "
            f"{secs(r['seconds_mean'])} | {pct(r['execution_accuracy'])} |"
        )
    return "\n".join(rows)


def tune_table(doc: dict[str, Any]) -> str:
    rows = ["| dev setting | execution acc. | p50 s |", "| --- | ---: | ---: |"]
    rows += [
        f"| {r['label']} | {pct(r['execution_accuracy'])} | {secs(r['seconds_p50'])} |"
        for r in doc["metrics"]["grid"]
    ]
    chosen = doc["metrics"]["chosen"]
    rows += [
        "",
        f"Chosen on dev: c_puct {chosen['c_puct']}, first-play urgency reduction "
        f"{chosen['fpu_reduction']}, prior weight {chosen['prior_weight']}.",
    ]
    return "\n".join(rows)


def train_lines(doc: dict[str, Any]) -> str:
    m = doc["metrics"]
    s = m["seconds"]
    return (
        f"Trained on {m['train_questions']:,} training questions "
        f"({m['gold_reachable']:,} with a reachable gold program) in {s['total'] / 60:.0f} min "
        f"with {doc['config']['workers']} workers; the ranker saw {m['ranker_programs']:,} "
        f"labelled programs ({pct(m['ranker_positive_rate'])} correct), "
        f"{m['ranker_programs_from_search']:,} of them from the second, MCTS round. "
        f"Model files: " + ", ".join(f"{k} {v:,} bytes" for k, v in m["model_bytes"].items()) + "."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    prefix = "smoke-" if parser.parse_args().smoke else ""
    sections: list[tuple[str, str]] = []
    if test := load(prefix, "test"):
        sections += [("curve", curve(test)), ("methods", method_table(test))]
        sections.append(("questions", f"Test questions: {questions(test)}."))
        sections.append(("reachable", reach_line(test)))
        sections.append(("measured", measured_on(test)))
    if doc := load(prefix, "compare"):
        sections.append(("compare", compare_table(doc)))
        sections.append(("ladder", ladder_table(doc)))
    if doc := load(prefix, "retrieval"):
        sections.append(("retrieval", retrieval_table(doc)))
        sections.append(("retrieval timing", retrieval_timing(doc)))
    if doc := load(prefix, "latency"):
        sections.append(("latency", latency_table(doc)))
        sections.append(("latency questions", f"Latency questions: {doc['config']['rule']}."))
    if doc := load(prefix, "tune"):
        sections.append(("tune", tune_table(doc)))
        sections.append(("tune questions", f"Dev questions: {questions(doc)}."))
    if doc := load(prefix, "train"):
        sections.append(("train", train_lines(doc)))
    if doc := load(prefix, "trace"):
        sections.append(("trace", "```mermaid\n" + doc["metrics"]["mermaid"] + "\n```"))
    for name, text in sections:
        print(f"<!-- {name} -->\n{text}\n")


if __name__ == "__main__":
    main()
