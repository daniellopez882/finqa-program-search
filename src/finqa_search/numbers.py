"""Numeric candidates: every number a program could use, with where it came from.

Numbers are read from sentences, table cells and the question. Each mention keeps the
context a learned prior needs: the row label and the normalised column header for a
cell, the nearby words for a sentence, the year the number belongs to, whether it was a
percentage and the unit scale ("in millions", "$ 3.8 billion"). Mentions with the same
value are merged into one candidate, because programs over equal values execute the same.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

from finqa_search.dataset import Example
from finqa_search.dsl import number_value, value_key
from finqa_search.retrieval.bm25 import tokenize

_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
# A FinQA number token: optional sign, digits with optional thousands commas, optional
# decimals, or a bare leading-dot decimal. Letters glued on ("10-k", "2i") disqualify it.
_NUM_TOKEN_RE = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?$|^-?\.\d+%?$")

UNIT_WORDS: dict[str, float] = {
    "thousand": 1e3,
    "thousands": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "mm": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "trillion": 1e12,
}

MONTHS = frozenset(
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sep sept oct nov dec".split()
)

CONTEXT_WINDOW = 6
ANCHOR_WINDOW = 4


def years_in(text: str) -> list[int]:
    return [int(y) for y in _YEAR_RE.findall(text)]


def unit_of(text: str) -> float:
    """The scale a header or caption declares, e.g. '( in millions )' -> 1e6."""
    low = text.lower()
    for word, scale in UNIT_WORDS.items():
        if re.search(rf"\b{word}\b", low):
            return scale
    return 1.0


def normalise_header(text: str) -> str:
    """Lower-case a column header and drop the noise that varies between reports."""
    low = text.lower().replace("$", " ")
    low = re.sub(r"\(\s*\d+\s*\)", " ", low)  # footnote markers like ( 1 )
    low = re.sub(r"\b(?:fiscal|year|years|ended|ending|as of|at)\b", " ", low)
    return re.sub(r"\s+", " ", low).strip(" ,.:;")


def is_year_value(value: float, text: str) -> bool:
    return value.is_integer() and 1950 <= value <= 2049 and "." not in text and "," not in text


@dataclass(frozen=True, slots=True)
class Mention:
    value: float
    text: str
    fact_id: str
    kind: str  # "text", "table" or "question"
    position: int
    row: int = -1
    col: int = -1
    row_label: str = ""
    col_header: str = ""
    year: int | None = None
    percent: bool = False
    unit: float = 1.0
    context: tuple[str, ...] = ()
    anchor: tuple[str, ...] = ()  # words that name the number: row label, or words before it
    negative_hint: bool = False  # written as "( 123 )" in running text
    variant: str = ""  # "", or "pct" / "abs" / "neg" for a derived reading


@dataclass(slots=True)
class Candidate:
    """All mentions of one value. ``text`` is how a program writes it."""

    value: float
    text: str
    mentions: list[Mention] = field(default_factory=list)

    @property
    def key(self) -> str:
        return value_key(self.value)


def _clean_number(token: str) -> tuple[float, str, bool] | None:
    tok = token.strip()
    if not _NUM_TOKEN_RE.match(tok):
        return None
    pct = tok.endswith("%")
    bare = tok.rstrip("%").replace(",", "")
    value = number_value(bare)
    if value is None or not math.isfinite(value):
        return None
    return value, bare, pct


def _after_month(tokens: list[str], i: int) -> bool:
    """Whether token ``i`` follows a month name, also when FinQA split "dec." into "dec ."."""
    j = i - 1
    if j >= 1 and tokens[j] == ".":
        j -= 1
    return j >= 0 and tokens[j].lower().rstrip(".") in MONTHS


def text_mentions(sentence: str, fact_id: str, kind: str, base_unit: float = 1.0) -> list[Mention]:
    """Numbers in one FinQA sentence (already split into space-separated tokens)."""
    tokens = sentence.split()
    out: list[Mention] = []
    seen: set[tuple[int, str]] = set()
    for i, tok in enumerate(tokens):
        parsed = _clean_number(tok.lstrip("$"))
        if parsed is None:
            continue
        value, bare, pct = parsed
        nxt = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
        if nxt == "%":
            pct = True
        prev = tokens[i - 1] if i else ""
        # "december 31 , 2009": a day of the month is part of a date, not a quantity. Most
        # FinQA pages have one, and gold programs almost never use it.
        if bare.isdigit() and 1 <= value <= 31 and _after_month(tokens, i):
            continue
        # FinQA writes "2.05% ( 2.05 % )": the bracketed echo is the same mention again.
        if prev == "(" and nxt == "%" and (i - 2, bare) in seen:
            continue
        seen.add((i, bare))
        unit = UNIT_WORDS.get(nxt, base_unit)
        bracketed = prev in ("(", "$(") or (prev == "$" and i >= 2 and tokens[i - 2] == "(")
        negative = bracketed and nxt == ")"
        lo, hi = max(0, i - CONTEXT_WINDOW), min(len(tokens), i + CONTEXT_WINDOW + 1)
        window = " ".join(tokens[lo:hi])
        before = " ".join(tokens[max(0, i - ANCHOR_WINDOW) : i])
        near_years = [y for y in years_in(window) if str(y) != bare]
        out.append(
            Mention(
                value=value,
                text=bare,
                fact_id=fact_id,
                kind=kind,
                position=i,
                year=near_years[0] if near_years else None,
                percent=pct,
                unit=unit,
                context=tuple(tokenize(window)),
                anchor=tuple(t for t in tokenize(before) if not t[0].isdigit()),
                negative_hint=negative,
            )
        )
    return out


def cell_number(cell: str) -> tuple[float, str, bool] | None:
    """A table cell the way FinQA's interpreter reads it: '$' dropped, '( .. )' echo cut."""
    head = cell.replace("$", " ").split("(")[0].strip()
    if not head:
        return None
    pct = head.endswith("%")
    parsed = _clean_number(head.replace(" ", ""))
    if parsed is None:
        return None
    value, bare, inner_pct = parsed
    return value, bare, pct or inner_pct


def table_mentions(table: list[list[str]]) -> list[Mention]:
    """Numbers in table cells.

    Row 0 is the header. FinQA's table extraction sometimes leaves the first data row up
    there ("$ 25621"), so numeric header cells are mentions too, filed under ``table_0``.
    """
    if not table:
        return []
    header = table[0]
    table_unit = unit_of(" ".join(header[:1]))
    out: list[Mention] = []
    for r in range(len(table)):
        row = table[r]
        label = row[0] if row else ""
        label_tokens = tuple(tokenize(label))
        for c in range(len(row)):
            parsed = cell_number(row[c])
            if c == 0 and (parsed is None or is_year_value(parsed[0], parsed[1])):
                continue  # a row label is a number only when the whole label is one
            if parsed is not None:
                readings = [(parsed[0], parsed[1], parsed[2], False)]
            else:
                # Not a clean numeric cell ("$ 10.8 million", "1.61% - 2.14%", "( $ 512 )"):
                # take every number token in it, as for a sentence.
                readings = [
                    (t.value, t.text, t.percent, t.negative_hint)
                    for t in text_mentions(row[c], f"table_{r}", "table")
                ]
            head = header[c] if 0 < c < len(header) and r > 0 else ""
            ys = years_in(head) or years_in(label)
            head_unit = unit_of(head)
            for value, bare, pct, negative in readings:
                if r == 0 and is_year_value(value, bare):
                    continue
                out.append(
                    Mention(
                        value=value,
                        text=bare,
                        fact_id=f"table_{r}",
                        kind="table",
                        position=c,
                        row=r,
                        col=c,
                        row_label=label,
                        col_header=normalise_header(head),
                        year=ys[0] if ys else None,
                        percent=pct or "%" in head,
                        unit=head_unit if head_unit != 1.0 else table_unit,
                        context=label_tokens + tuple(tokenize(head)),
                        anchor=label_tokens,
                        negative_hint=negative,
                    )
                )
    return out


def variants(m: Mention) -> list[Mention]:
    """Readings a gold program may use instead of the literal number.

    Annotators wrote "2%" (0.02) for a percentage as often as "2", and used the magnitude
    of a bracketed negative cell as often as its sign.
    """
    out: list[Mention] = []
    if m.percent:
        out.append(replace(m, value=m.value / 100.0, text=f"{m.text}%", variant="pct"))
    if m.value < 0:
        out.append(replace(m, value=-m.value, text=m.text.lstrip("-"), variant="abs"))
    elif m.negative_hint:
        out.append(replace(m, value=-m.value, text=f"-{m.text}", variant="neg"))
    return out


def example_mentions(
    example: Example, fact_ids: set[str] | None = None, with_variants: bool = True
) -> list[Mention]:
    """Mentions from the question and from the facts in ``fact_ids`` (all facts if None).

    Header-row numbers come along whenever any table row is wanted.
    """
    out = text_mentions(example.question, "question", "question")
    for i, sentence in enumerate(example.sentences):
        fid = f"text_{i}"
        if fact_ids is None or fid in fact_ids:
            out.extend(text_mentions(sentence, fid, "text"))
    any_table = fact_ids is None or any(f.startswith("table_") for f in fact_ids)
    out.extend(
        m
        for m in table_mentions(example.table)
        if fact_ids is None or m.fact_id in fact_ids or (m.fact_id == "table_0" and any_table)
    )
    if with_variants:
        out.extend(v for m in list(out) for v in variants(m))
    return out


def merge(mentions: list[Mention]) -> list[Candidate]:
    """Group mentions by value, keeping first-seen order."""
    by_key: dict[str, Candidate] = {}
    for m in mentions:
        key = value_key(m.value)
        cand = by_key.get(key)
        if cand is None:
            by_key[key] = Candidate(value=m.value, text=m.text, mentions=[m])
        else:
            cand.mentions.append(m)
    return list(by_key.values())
