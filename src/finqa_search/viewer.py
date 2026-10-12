"""A self-contained HTML page that draws one search trace as a collapsible tree."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Search tree: __TITLE__</title>
<style>
:root { --bg:#fbfbf8; --fg:#1d1d1b; --muted:#6b6b66; --line:#d9d8d2; --visit:#3b6ea8;
        --prior:#c48a2c; --good:#2e7d4f; --bad:#b4423a; --chip:#efeee9; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161615; --fg:#e8e7e2; --muted:#9a9993; --line:#3a3936; --visit:#7aa7da;
          --prior:#e0b260; --good:#6cc28f; --bad:#e4766d; --chip:#262624; }
}
body { background:var(--bg); color:var(--fg); margin:0; padding:16px;
       font:14px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
header { max-width:1100px; margin:0 auto 12px; }
h1 { font-size:16px; margin:0 0 6px; }
.meta { color:var(--muted); }
.meta b { color:var(--fg); font-weight:600; }
.legend span { display:inline-block; margin-right:14px; }
.bar { display:inline-block; height:8px; vertical-align:middle; border-radius:2px; }
main { max-width:1100px; margin:0 auto; overflow-x:auto; }
ul { list-style:none; margin:0; padding-left:18px; border-left:1px solid var(--line); }
li { margin:3px 0; }
details > summary { cursor:pointer; list-style:none; }
details > summary::-webkit-details-marker { display:none; }
.node { display:inline-flex; gap:10px; align-items:center; flex-wrap:wrap; }
.edge { color:var(--muted); }
.prog { background:var(--chip); padding:1px 6px; border-radius:4px; }
.stop { color:var(--muted); }
.good { color:var(--good); font-weight:600; }
.bad { color:var(--bad); }
.num { color:var(--muted); }
.unvisited { opacity:.55; }
.legend .vis { width:24px; background:var(--visit); }
.legend .pri { width:24px; background:var(--prior); }
section.evidence { max-width:1100px; margin:16px auto 0; color:var(--muted); }
section.evidence h2 { font-size:14px; color:var(--fg); margin:12px 0 4px; }
section.evidence li { margin:2px 0; }
section.evidence ul { border-left:none; padding-left:0; }
.gold { color:var(--good); }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="meta">__META__</div>
  <div class="meta legend">
    <span><span class="bar vis"></span> visits</span>
    <span><span class="bar pri"></span> prior of the edge</span>
    <span>Q = mean backed-up value, max = best value below,
      stop = ranker score of answering here</span>
  </div>
</header>
<main id="tree"></main>
<section class="evidence">
  <h2>Retrieved facts (best first; gold facts in green)</h2>
  <ul id="facts"></ul>
  <h2>Leaves the search could use</h2>
  <ul id="leaves"></ul>
</section>
<script id="trace" type="application/json">__DATA__</script>
<script>
(() => {
const data = JSON.parse(document.getElementById("trace").textContent);
const byId = new Map(data.trace.nodes.map(n => [n.id, n]));
const rootVisits = Math.max(1, (byId.get(data.trace.root) || {visits: 1}).visits);
const best = data.program;
function fmt(x) { return x === null || x === undefined ? "-" : Number(x).toFixed(3); }
function bar(width, colour) {
  const s = document.createElement("span");
  s.className = "bar"; s.style.width = Math.max(1, Math.round(width)) + "px";
  s.style.background = colour; return s;
}
function label(node) {
  const wrap = document.createElement("span"); wrap.className = "node";
  const prog = document.createElement("span"); prog.className = "prog";
  prog.textContent = node.program;
  if (node.terminal && best && node.program === best + " -> stop") {
    prog.classList.add(data.exe_correct ? "good" : "bad");
  }
  wrap.append(prog, bar(80 * node.visits / rootVisits, "var(--visit)"));
  const nums = document.createElement("span"); nums.className = "num";
  nums.textContent = `N=${node.visits}  Q=${fmt(node.mean_value)}  ` +
    `max=${fmt(node.max_value)}  stop=${fmt(node.stop_score)}`;
  wrap.append(nums); return wrap;
}
function render(node, depth) {
  const li = document.createElement("li");
  const edges = node.edges || [];
  if (!edges.length) { li.append(label(node)); return li; }
  const det = document.createElement("details"); det.open = depth < 2;
  const sum = document.createElement("summary"); sum.append(label(node)); det.append(sum);
  const ul = document.createElement("ul");
  for (const e of edges) {
    const item = document.createElement("li");
    const head = document.createElement("span"); head.className = "edge";
    head.append(bar(60 * e.prior, "var(--prior)"));
    head.append(` ${e.action}  P=${fmt(e.prior)}  n=${e.visits}`);
    if (e.child === null || !byId.has(e.child)) {
      item.className = "unvisited"; item.append(head);
    } else {
      const sub = render(byId.get(e.child), depth + 1);
      item.append(head, sub);
    }
    ul.append(item);
  }
  det.append(ul); li.append(det); return li;
}
const root = byId.get(data.trace.root);
const rootList = document.createElement("ul"); rootList.style.borderLeft = "none";
if (root) rootList.append(render(root, 0));
document.getElementById("tree").append(rootList);
const gold = new Set(data.gold_facts || []);
for (const f of data.retrieved_facts || []) {
  const li = document.createElement("li");
  li.textContent = `${f.id}: ${f.text}`;
  if (gold.has(f.id)) li.className = "gold";
  document.getElementById("facts").append(li);
}
for (const leaf of data.leaves || []) {
  const li = document.createElement("li");
  li.textContent = `${leaf.arg} = ${leaf.value}  (${leaf.source})`;
  document.getElementById("leaves").append(li);
}
})();
</script>
</body>
</html>
"""


def render_html(trace: dict[str, Any]) -> str:
    """``trace`` is one search-trace record (see ``finqa-search solve --trace``)."""
    title = html.escape(trace.get("question", trace.get("id", "search trace")))
    bits = [
        f"<b>{html.escape(str(trace.get('method', '')))}</b>",
        f"answer <b>{html.escape(str(trace.get('answer')))}</b>",
        f"gold <b>{html.escape(str(trace.get('gold_answer')))}</b>",
        f"program <b>{html.escape(str(trace.get('program')))}</b>",
        f"gold program <b>{html.escape(str(trace.get('gold_program')))}</b>",
        f"{trace.get('simulations', 0)} simulations, {trace.get('nodes', 0)} nodes, "
        f"{float(trace.get('seconds', 0.0)):.2f} s",
    ]
    data = json.dumps(trace).replace("</", "<\\/")
    return (
        _PAGE.replace("__TITLE__", title)
        .replace("__META__", " &middot; ".join(bits))
        .replace("__DATA__", data)
    )


def write_html(trace: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(trace), encoding="utf-8")
    return path


def mermaid(record: dict[str, Any], max_children: int = 3, max_depth: int = 3) -> str:
    """The most visited part of a record's search tree as a Mermaid flowchart (for the README)."""
    tree = record["trace"]
    by_id = {n["id"]: n for n in tree["nodes"]}
    lines = ["flowchart LR"]

    def esc(text: str) -> str:
        return text.replace('"', "'")

    def visit(node_id: int, depth: int) -> None:
        node = by_id[node_id]
        stop = node.get("stop_score")
        extra = f"<br/>stop={stop:.3f}" if isinstance(stop, float) else ""
        lines.append(
            f'  n{node_id}["{esc(node["program"])}<br/>N={node["visits"]} '
            f'Q={node["mean_value"] or 0:.3f}{extra}"]'
        )
        if depth >= max_depth:
            return
        edges = [e for e in node["edges"] if e["child"] is not None and e["child"] in by_id]
        edges.sort(key=lambda e: -e["visits"])
        for e in edges[:max_children]:
            lines.append(f'  n{node_id} -->|"P={e["prior"]:.2f} n={e["visits"]}"| n{e["child"]}')
            visit(e["child"], depth + 1)

    if tree.get("root") is not None:
        visit(tree["root"], 0)
    return "\n".join(lines)
