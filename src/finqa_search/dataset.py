"""FinQA examples and the facts (sentences and table rows) a question is grounded in.

Only the report text, the cleaned table and the question are used as model input. The
release also ships the outputs of the FinQANet retriever (``table_retrieved``,
``text_retrieved``, ``model_input``, ``tfidftopn``); those are never read, because they
come from a trained retriever and would leak into this one's evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from finqa_search.dsl import Program, ProgramError, parse_program
from finqa_search.sources import data_dir

Split = Literal["train", "dev", "test"]
SPLITS: tuple[Split, ...] = ("train", "dev", "test")


class Fact(BaseModel):
    """One retrievable unit: a sentence (``text_i``) or a linearised table row (``table_i``)."""

    model_config = ConfigDict(frozen=True)

    id: str
    kind: Literal["text", "table"]
    text: str
    index: int


class Example(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    filename: str
    question: str
    pre_text: list[str]
    post_text: list[str]
    table: list[list[str]]
    program: str
    exe_ans: float | str
    answer: str = ""
    gold_inds: dict[str, str] = Field(default_factory=dict)

    @property
    def sentences(self) -> list[str]:
        return [*self.pre_text, *self.post_text]

    def gold_program(self) -> Program | None:
        try:
            return parse_program(self.program)
        except ProgramError:
            return None

    def facts(self) -> list[Fact]:
        out = [
            Fact(id=f"text_{i}", kind="text", text=s, index=i)
            for i, s in enumerate(self.sentences)
            if len(s.strip(" .")) > 1
        ]
        out.extend(
            Fact(id=f"table_{i}", kind="table", text=linearise_row(self.table, i), index=i)
            for i in range(1, len(self.table))
        )
        return out


def linearise_row(table: list[list[str]], row: int) -> str:
    """'the <column header> of <row label> is <cell> ;' for every cell of a row."""
    header = table[0] if table else []
    cells = table[row]
    label = cells[0] if cells else ""
    parts: list[str] = []
    for col in range(1, len(cells)):
        head = header[col] if col < len(header) else ""
        parts.append(f"the {head} of {label} is {cells[col]} ;")
    return " ".join(parts) if parts else label


def from_raw(raw: dict) -> Example:
    qa = raw["qa"]
    return Example(
        id=raw["id"],
        filename=raw.get("filename", ""),
        question=qa["question"],
        pre_text=list(raw.get("pre_text", [])),
        post_text=list(raw.get("post_text", [])),
        table=[list(map(str, row)) for row in raw.get("table", [])],
        program=qa.get("program", ""),
        exe_ans=qa.get("exe_ans", ""),
        answer=str(qa.get("answer", "")),
        gold_inds=dict(qa.get("gold_inds", {})),
    )


def split_path(split: Split, root: Path | None = None) -> Path:
    return (root or data_dir()) / "finqa" / f"{split}.json"


def load_split(split: Split, root: Path | None = None) -> list[Example]:
    with split_path(split, root).open(encoding="utf-8") as fh:
        return [from_raw(r) for r in json.load(fh)]


def report_of(example_id: str) -> str:
    """The report page an example comes from: ``ADI/2009/page_49.pdf-1`` -> its PDF page."""
    return example_id.rsplit("-", 1)[0]


def leakage(splits: dict[str, list[Example]]) -> dict[str, int]:
    """How many example ids and report pages two splits share (FinQA splits by question)."""
    out: dict[str, int] = {}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ids_a = {e.id for e in splits[a]}
            ids_b = {e.id for e in splits[b]}
            pages_a = {report_of(e.id) for e in splits[a]}
            pages_b = {report_of(e.id) for e in splits[b]}
            qs_a = {(report_of(e.id), e.question.strip().lower()) for e in splits[a]}
            qs_b = {(report_of(e.id), e.question.strip().lower()) for e in splits[b]}
            out[f"{a}/{b} ids"] = len(ids_a & ids_b)
            out[f"{a}/{b} pages"] = len(pages_a & pages_b)
            out[f"{a}/{b} same page and question"] = len(qs_a & qs_b)
    return out
