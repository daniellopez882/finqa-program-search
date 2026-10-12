"""Okapi BM25 over the facts of one report page, with IDF learned on the training corpus.

A single page has only a few dozen facts, so document frequencies computed on it alone
are noise. IDF is fitted once over every training fact and then reused for each page;
term frequencies and lengths still come from the page being searched.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?")

STOPWORDS = frozenset(
    (
        "a an and are as at be by for from has have in is it its of on or that the this to "
        "was were what which with how much many did does do during than then there their "
        "these those our we us"
    ).split()
)


def stem(token: str) -> str:
    """Strip the commonest English inflections; enough to match 'revenues' with 'revenue'."""
    if token.isdigit() or len(token) <= 3:
        return token
    for suffix, keep in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "s" and token.endswith(("ss", "us", "is")):
                return token
            return token[: -len(suffix)] + keep
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class IdfTable:
    """Document frequencies over a reference corpus; unseen terms get the maximum IDF."""

    n_docs: int = 0
    df: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, docs: Iterable[str]) -> IdfTable:
        df: Counter[str] = Counter()
        n = 0
        for doc in docs:
            df.update(set(tokenize(doc)))
            n += 1
        return cls(n_docs=n, df=dict(df))

    def idf(self, term: str) -> float:
        d = self.df.get(term, 0)
        # The BM25+ style floor keeps very common terms from going negative.
        return math.log(1.0 + (self.n_docs - d + 0.5) / (d + 0.5))


@dataclass
class BM25:
    idf: IdfTable
    k1: float = 1.2
    b: float = 0.75

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        q_terms = tokenize(query)
        doc_tokens = [tokenize(d) for d in docs]
        if not doc_tokens:
            return []
        avgdl = sum(len(t) for t in doc_tokens) / len(doc_tokens) or 1.0
        weights = {t: self.idf.idf(t) for t in set(q_terms)}
        q_counts = Counter(q_terms)
        out: list[float] = []
        for toks in doc_tokens:
            tf = Counter(toks)
            norm = self.k1 * (1.0 - self.b + self.b * len(toks) / avgdl)
            s = 0.0
            for term, qc in q_counts.items():
                f = tf.get(term, 0)
                if f:
                    s += qc * weights[term] * f * (self.k1 + 1.0) / (f + norm)
            out.append(s)
        return out
