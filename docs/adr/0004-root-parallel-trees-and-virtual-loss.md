# ADR 0004 — Root parallelism across processes, virtual loss inside a tree

| Date | Status |
| --- | --- |
| 2026-10-11 | accepted |

## Situation

The reference machine has six cores and runs Windows, where Python worker processes
are started with `spawn` and nothing is shared between them unless it is explicitly
put in shared memory. The cost of one simulation is dominated by two batched model
calls: the LightGBM prior over every legal step of a new node, and the LightGBM ranker
over the programs that node can finish. Both are much cheaper per row when many rows
go in one call.

Two kinds of parallelism matter:

* across questions, for throughput over a whole split; and
* inside one question, for the latency of a single answer.

## Choice

* **Across questions:** a process pool, one question per task, each worker loading the
  prior and the ranker once in its initializer.
* **Inside a tree:** leaves are collected `batch` (8) at a time. Each selected path gets
  a virtual loss (a pending visit that counts towards `N` with no value) so the next
  descent in the same batch prefers other branches. The batch's new nodes are expanded
  with one prior call and evaluated with one ranker call, then the virtual losses are
  removed and the real values backed up.
* **Inside a question, across cores:** root parallelism. `search/parallel.py` runs
  `trees` independent trees with seeds `seed + i`, splits the simulation budget evenly,
  and adds Dirichlet noise (weight 0.25) to the root prior of every tree but the first.
  The merge takes the best-scoring program over all trees, which is the same rule a
  single tree uses, and sums root visit counts for the trace. The same function runs
  the trees one after another in-process (`Method(trees=k)`) or in a pool.

## Consequences

* No locks, no shared-memory tree: a worker is the single-tree code with a different
  seed and budget.
* Each tree rebuilds the question's context and re-discovers transpositions the others
  already found, so `k` trees with `n / k` simulations each explore less deeply than one
  tree with `n`. Whether the extra breadth pays for that is measured, not assumed: the
  latency benchmark reports accuracy as well as p50/p95 seconds for one and six workers.
* Batched evaluation means a selected leaf's value arrives up to `batch - 1` descents
  late, so the search is not identical to sequential PUCT with the same number of
  simulations. It stays deterministic for a fixed seed and batch size.

## Rejected alternatives

* **Tree parallelism with a shared tree and locks.** Python threads would serialise on
  the interpreter for everything except the LightGBM calls, and a tree in shared memory
  across `spawn`ed processes means serialising nodes on every access.
* **Leaf parallelism (several rollouts from one leaf).** There are no random rollouts to
  parallelise: a leaf is valued by the ranker directly, and batching already amortises
  that call.
* **Threads inside the LightGBM calls.** LightGBM can use several threads per predict
  call, but the question-level pool already keeps every core busy during a split run,
  and oversubscribing cores makes per-question timings depend on what else is running.
  Every predict call is pinned to one thread.
