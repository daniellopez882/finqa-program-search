"""Where each stage reads and writes inside the data directory."""

from __future__ import annotations

from pathlib import Path

from finqa_search.sources import data_dir


def retrieval_dir(root: Path | None = None) -> Path:
    return (root or data_dir()) / "retrieval"


def idf_path(root: Path | None = None) -> Path:
    return retrieval_dir(root) / "idf.json"


def rankings_path(split: str, root: Path | None = None) -> Path:
    return retrieval_dir(root) / f"{split}.jsonl"


def models_dir(root: Path | None = None) -> Path:
    return (root or data_dir()) / "models"


def runs_dir(root: Path | None = None) -> Path:
    return (root or data_dir()) / "runs"
