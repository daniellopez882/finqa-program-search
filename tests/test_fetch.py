"""Checksum handling of the downloader, using file:// URLs instead of the network."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from finqa_search.fetch import ChecksumError, ensure_file, sha256_of
from finqa_search.sources import CROSS_ENCODER, DATA_DIR_ENV, FINQA_FILES, data_dir


def test_ensure_file_downloads_and_verifies(tmp_path: Path) -> None:
    src = tmp_path / "src.json"
    src.write_bytes(b'{"a": 1}')
    digest = hashlib.sha256(b'{"a": 1}').hexdigest()
    dest = tmp_path / "out" / "a.json"
    logs: list[str] = []
    ensure_file(src.as_uri(), dest, digest, logs.append)
    assert dest.read_bytes() == b'{"a": 1}'
    assert sha256_of(dest) == digest
    assert logs[-1].startswith("verified")
    ensure_file(src.as_uri(), dest, digest, logs.append)
    assert logs[-1].startswith("ok")


def test_a_wrong_checksum_leaves_nothing_behind(tmp_path: Path) -> None:
    src = tmp_path / "src.json"
    src.write_bytes(b"tampered")
    dest = tmp_path / "out" / "a.json"
    with pytest.raises(ChecksumError, match="does not match"):
        ensure_file(src.as_uri(), dest, "0" * 64, lambda _: None)
    assert not dest.exists()
    assert not dest.with_name("a.json.part").exists()


def test_a_corrupt_local_copy_is_downloaded_again(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"good")
    dest = tmp_path / "a.bin"
    dest.write_bytes(b"bad")
    logs: list[str] = []
    ensure_file(src.as_uri(), dest, hashlib.sha256(b"good").hexdigest(), logs.append)
    assert dest.read_bytes() == b"good"
    assert logs[0].startswith("mismatch")


def test_data_dir_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert data_dir() == tmp_path
    monkeypatch.delenv(DATA_DIR_ENV)
    assert data_dir() == Path.home() / ".cache" / "finqa-program-search"


def test_every_remote_file_is_pinned() -> None:
    for f in FINQA_FILES:
        assert len(f.sha256) == 64
        assert "/0f16e2867befa6840783e58be38c9efb9229d742/" in f.url
    assert set(CROSS_ENCODER.sha256) == {CROSS_ENCODER.onnx_file, "tokenizer.json", "config.json"}
    assert CROSS_ENCODER.revision in CROSS_ENCODER.url("config.json")
