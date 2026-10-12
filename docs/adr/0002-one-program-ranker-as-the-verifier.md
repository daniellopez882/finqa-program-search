# ADR 0002 — A program-only ranker steers the search; ranker and prior pick the answer

| Date | Status |
| --- | --- |
| 2026-10-12 | accepted |

## Situation

PUCT needs a value for every node it creates, and at the end something has to choose
one finished program as the answer. FinQA gives the gold answer for training questions
only, so at test time the only way to judge a program is a learned model of "this
program answers this question". Three choices were open:

1. **What the value model sees.** Either the finished program and its grounding (which
   numbers, from which facts, what it evaluates to), or also how the search got there
   (visit counts, path probabilities).
2. **How a half-built program is valued.** Most good programs are two steps long and
   their first step (`subtract(2009 value, 2008 value)`) is a wrong answer on its own.
3. **What picks the final answer.**

The failure mode to design against is the usual one for search over a learned
verifier: a search that maximises the ranker visits far more programs than the
candidates the ranker was trained on, and the more programs it scores, the more it finds
that the ranker over-rates. Greedy decoding never meets those programs, because it only
follows the prior. So a search can score a correct program and still answer with a
wrong one that the ranker happens to like, and that gap can grow with the simulation
budget.

## Choice

* `ranker.py` is a LightGBM binary classifier over `PROGRAM_FEATURES`: what the program
  computes, which leaves it uses and where they came from, how its numbers relate to
  the question. It never sees search statistics, so greedy, beam, the exhaustive depth-2
  baseline and MCTS are all judged by the same model.
* The ranker is trained on programs labelled by execution against `exe_ans`, generated
  for each training question with a prior that was fitted on other report pages (3-fold
  cross-fitting by page), so its inputs look like test-time candidates. A first round
  draws them from beam search, ancestral samples of the prior and uniform depth-2
  draws. A second round fits one ranker per fold on the first round, runs MCTS on each
  training question with the prior and ranker of the fold that excludes it, and adds
  every program that search scored: exactly the programs a search finds attractive,
  which the first round under-represents. The final ranker is fitted on both rounds.
* A new node's value is the best ranker probability among stopping now and stopping
  after one of its three most probable next steps (`lookahead = 3`). This one-step
  lookahead keeps a correct first step from scoring like a wrong answer.
* The search maximises the ranker. The answer is the finished program with the highest
  `log P_ranker(correct) + w * log P_prior(program)`, where the prior term is the
  log-likelihood of the most probable derivation of that program the search saw. `w`
  is chosen on dev (`bench/run.py --stages tune`), from a grid that includes `w = 0`,
  the ranker alone, so the choice of rule is a measured one. Beam search with
  re-ranking uses the same rule; the exhaustive depth-2 baseline has no prior and uses
  the ranker alone.

## Consequences

* One verifier for every method makes the accuracy-against-compute comparison fair:
  differences come from which programs a method finds, not from how it scores them.
* "A correct program was scored" (candidate coverage) and "the chosen program is
  correct" are reported separately for every method and budget. Their gap is the
  verifier's error.
* Values and the answer rule differ on purpose: exploring by ranker probability alone
  still finds programs the prior under-rates, and the prior term then guards the final
  choice. The cost is that the tree statistics (visits, Q) do not show the score that
  picked the answer; the trace records both.
* Training the ranker needs a search per training question (the second round), which
  is the slowest part of `finqa-search train`.

## Rejected alternatives

* **Prior likelihood as the value** (no ranker). This is what beam search optimises; it
  cannot use execution results (sign, magnitude, yes/no) or how well a program's numbers
  match the question.
* **The ranker alone picks the answer.** Over-trusts a model that the search has
  optimised against. It stays in the dev grid as `w = 0`; if it wins there, the tuned
  setting uses it.
* **Prior log-likelihood as a ranker feature.** Would let LightGBM learn the weight
  instead of tuning one number, but the exhaustive baseline scores every depth-2
  program of a question with no prior, and the uniform draws in the ranker's
  training data would need one extra prior call each.
* **Random rollouts to a finished program.** Most random completions are nonsense
  programs, so rollout values are noise next to a ranker score.
* **Only the first round of ranker candidates.** Cheaper to train, but the ranker
  then never sees the programs a ranker-maximising search drifts towards, which are
  the ones it has to reject at test time.
