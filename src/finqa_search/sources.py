"""Where the data and the cross-encoder come from, and the sha256 each file must hash to.

FinQA is pinned to a commit of the official repository (czyssrs/FinQA, MIT). The
cross-encoder is the int8 ONNX export of cross-encoder/ms-marco-MiniLM-L6-v2 (Apache-2.0)
at a fixed Hub revision. Checksums were recorded from the first download on 2026-10-11.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DATA_DIR_ENV = "FINQA_SEARCH_DATA_DIR"

FINQA_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
FINQA_BASE_URL = f"https://raw.githubusercontent.com/czyssrs/FinQA/{FINQA_COMMIT}"


def data_dir() -> Path:
    """Root for FinQA, retrieval caches, trained models and run outputs."""
    raw = os.environ.get(DATA_DIR_ENV)
    root = Path(raw) if raw else Path.home() / ".cache" / "finqa-program-search"
    return root.expanduser()


def hf_home() -> Path:
    raw = os.environ.get("HF_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".cache" / "huggingface"


@dataclass(frozen=True)
class RemoteFile:
    url: str
    sha256: str
    path: str  # relative to the data directory


FINQA_FILES: tuple[RemoteFile, ...] = (
    RemoteFile(
        url=f"{FINQA_BASE_URL}/dataset/train.json",
        sha256="49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6",
        path="finqa/train.json",
    ),
    RemoteFile(
        url=f"{FINQA_BASE_URL}/dataset/dev.json",
        sha256="a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51",
        path="finqa/dev.json",
    ),
    RemoteFile(
        url=f"{FINQA_BASE_URL}/dataset/test.json",
        sha256="831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc",
        path="finqa/test.json",
    ),
    RemoteFile(
        url=f"{FINQA_BASE_URL}/LICENSE",
        sha256="92c0e67bd762c15c07ba60de09558338315a8ebf99aac4f51f487179809ac087",
        path="finqa/LICENSE",
    ),
)


@dataclass(frozen=True)
class HubModel:
    repo_id: str
    revision: str
    onnx_file: str
    licence: str
    sha256: dict[str, str]

    def snapshot_dir(self) -> Path:
        """The Hugging Face cache layout, so a copy fetched by another tool is reused."""
        folder = "models--" + self.repo_id.replace("/", "--")
        return hf_home() / "hub" / folder / "snapshots" / self.revision

    def url(self, filename: str) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{filename}"


CROSS_ENCODER = HubModel(
    repo_id="cross-encoder/ms-marco-MiniLM-L6-v2",
    revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
    onnx_file="onnx/model_quint8_avx2.onnx",
    licence="Apache-2.0",
    sha256={
        "onnx/model_quint8_avx2.onnx": (
            "c80a8b34256ea453093d612e3ac48d3d965a0c0a48c7906709af8b8e28461bf9"
        ),
        "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        "config.json": "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc",
    },
)
