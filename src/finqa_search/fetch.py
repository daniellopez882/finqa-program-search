"""Download FinQA and the cross-encoder, verifying every file against a pinned sha256."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

from finqa_search.sources import CROSS_ENCODER, FINQA_FILES, HubModel, RemoteFile, data_dir

CHUNK = 1 << 20


class ChecksumError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "finqa-program-search/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as out:
        shutil.copyfileobj(resp, out, CHUNK)
    got = sha256_of(part)
    if got != expected:
        part.unlink(missing_ok=True)
        raise ChecksumError(f"{url}: sha256 {got} does not match the pinned {expected}")
    part.replace(dest)


def ensure_file(url: str, dest: Path, expected: str, log: Callable[[str], None]) -> Path:
    if dest.is_file():
        got = sha256_of(dest)
        if got == expected:
            log(f"ok        {dest}")
            return dest
        log(f"mismatch  {dest} (sha256 {got[:12]}..), downloading again")
    log(f"download  {url}")
    _download(url, dest, expected)
    log(f"verified  {dest}")
    return dest


def fetch_finqa(log: Callable[[str], None] = print, root: Path | None = None) -> list[Path]:
    base = root or data_dir()
    return [ensure_file(f.url, base / f.path, f.sha256, log) for f in FINQA_FILES]


def fetch_model(model: HubModel = CROSS_ENCODER, log: Callable[[str], None] = print) -> Path:
    snap = model.snapshot_dir()
    for name, digest in model.sha256.items():
        ensure_file(model.url(name), snap / name, digest, log)
    return snap


def model_ready(model: HubModel = CROSS_ENCODER) -> bool:
    snap = model.snapshot_dir()
    return all((snap / name).is_file() for name in model.sha256)


def finqa_ready(root: Path | None = None) -> bool:
    base = root or data_dir()
    return all((base / f.path).is_file() for f in FINQA_FILES)


def verify_all(log: Callable[[str], None] = print) -> bool:
    """Re-hash everything already on disk without downloading."""
    good = True
    files: list[tuple[Path, str]] = [(data_dir() / f.path, f.sha256) for f in FINQA_FILES]
    snap = CROSS_ENCODER.snapshot_dir()
    files += [(snap / n, d) for n, d in CROSS_ENCODER.sha256.items()]
    for path, digest in files:
        if not path.is_file():
            log(f"missing   {path}")
            good = False
        elif sha256_of(path) != digest:
            log(f"mismatch  {path}")
            good = False
        else:
            log(f"ok        {path}")
    return good


__all__ = [
    "ChecksumError",
    "RemoteFile",
    "fetch_finqa",
    "fetch_model",
    "finqa_ready",
    "model_ready",
    "sha256_of",
    "verify_all",
]
