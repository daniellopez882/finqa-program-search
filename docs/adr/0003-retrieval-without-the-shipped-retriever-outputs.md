# ADR 0003 — Ground programs in our own fact ranking, cached per split

| Date | Status |
| --- | --- |
| 2026-10-11 | accepted |

## Situation

A FinQA page has on average a few dozen sentences and table rows, and the numbers a
program may use come from them. Offering every number on the page to the search makes
the first-step branching factor grow with the square of the page's numbers; offering
too few makes the gold program unreachable. Something has to pick the facts.

The FinQA release ships helpful-looking fields next to each question:
`table_retrieved`, `text_retrieved`, `model_input` and `tfidftopn`. They are the output
of FinQANet's retriever, which was trained on the gold fact indices of the training
split. Using them would import a trained component this project does not build and
whose errors are correlated with the labels.

## Choice

* Facts are the page's sentences (`text_i`) and its table rows linearised as
  "the *column* of *row* is *cell* ;" (`table_i`); ids match `gold_inds`, so Recall@k
  is measured against the annotation directly.
* BM25 (implemented in `retrieval/bm25.py`) ranks every fact. IDF is fitted once on the
  facts of the distinct training pages, because a single page is too small to estimate
  document frequencies and dev/test text must not leak into it.
* The int8 ONNX export of `cross-encoder/ms-marco-MiniLM-L6-v2` re-scores the BM25 top
  12. The final score inside that pool is `z(cross-encoder) + 0.3 * z(BM25)`, z-scored per
  question; facts outside the pool keep BM25 order below it.
* Rankings for all three splits are computed once (`finqa-search retrieve`) and cached
  as JSON lines under the data directory. Every later stage, training included, reads
  the cache.
* `dataset.py` never reads the shipped retriever fields; the loader drops them.

## Consequences

* Retrieval errors are this project's own and are reported (Recall@3 against the gold
  indices, BM25 alone and hybrid) next to the search results.
* The cross-encoder is the most expensive step per question, which is why it runs once
  per split and not inside the search loop.
* Training sees rankings produced the same way as test rankings, so the prior learns
  from the same kind of imperfect evidence it gets at test time.
* The ms-marco model was trained on web search passages, not on financial tables; no
  fine-tuning is done here, so its ordering of table rows is whatever transfers.

## Rejected alternatives

* **The shipped `table_retrieved` / `text_retrieved` fields.** Leakage of a trained
  component, as above.
* **Gold facts at test time.** Common in ablations but measures a different task.
* **Dense bi-encoder retrieval.** Another model download and an index for pages that
  have a few dozen facts; BM25 plus a re-ranker over a short pool covers that size.
* **Cross-encoder over every fact.** Encoder calls would grow with page length, spent
  on facts BM25 already puts far down; the pool size is a single constant to revisit if
  Recall@3 shows facts lost outside it.
