# ADR 0001 — Search over a typed step space, not over program text

| Date | Status |
| --- | --- |
| 2026-10-11 | accepted |

## Situation

A FinQA answer is a short program such as `subtract(5829, 5735), divide(#0, 5735)`. A
search needs a set of moves out of every partial program. The two obvious shapes are:

* token by token over the program text, the way a seq2seq decoder (FinQANet, an LLM)
  writes it; or
* step by step, where one move appends a whole `op(arg, arg)` built from arguments the
  question context can supply.

Token-level moves make most of the tree syntactically or semantically dead
(`divide(#3,` when there is no `#3`), and the same program appears at many leaves
because `add(a, b)` and `add(b, a)` are different token strings. For a tree search
whose cost is counted in simulations, every dead or duplicate leaf is budget wasted.

## Choice

`space.py` defines one move as one step whose arguments are indices into a per-question
list of leaves (numbers read from the top-ranked facts, then constants), earlier step
results, or retrieved table rows. Typing rules are applied when the legal moves are
generated, so illegal programs never enter the tree:

* constants only fill the (operation, slot) pairs seen in training programs
  (`CONSTANT_SLOTS`), and two constants never meet in one step;
* `exp` needs a step result as its base; `greater` must be the final step;
* a program may stop only when every earlier result feeds a later step, and a move is
  legal only if the steps left can still consume every dangling result;
* `add` and `multiply` are generated in a single argument order.

Each state carries a key made of the sorted canonical forms of its steps plus the form
of its last step. Two orders of independent steps that end in the same final
expression get the same key, so the MCTS transposition table merges them into one node.

## Consequences

* Every leaf of the tree is a program the interpreter can run, so every simulation ends
  in something the ranker can score.
* The space is closed: a gold program that needs a number outside the top facts, a
  constant outside `CONSTANT_SLOTS` or more than four steps cannot be produced. The share
  of questions whose gold program is reachable is measured and reported next to every
  accuracy figure, so this ceiling is visible rather than hidden.
* Feature rows for the leaf-only moves are built once per question and reused by every
  state, which keeps the prior's cost per expansion to one batched LightGBM call.
* Adding an operation means adding a typing rule and a feature id; there is no grammar
  to keep in sync.

## Rejected alternatives

* **Token-level search over program text.** Needs a learned decoder to be useful at
  all, wastes most expansions on dead prefixes and cannot share nodes between
  commuted programs.
* **Expression trees with free reassociation** (treating `(a + b) + c` and `a + (b + c)`
  as one node). Makes the transposition table stronger but the key computation needs a
  normal form for arbitrary rational expressions per expansion; the cheap per-step
  canonical form catches the common case, commuted operands and reordered independent
  steps.
* **No typing, filter afterwards.** Grows the branching factor at every depth with
  programs (two constants, `exp` of a raw number, a yes/no fed into arithmetic) that the
  prior and the ranker would have to learn to reject.
