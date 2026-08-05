from __future__ import annotations

from pathlib import Path

import pytest

from vla_sim.artifact_identity import sha256_directory, sha256_file


def test_file_hash_is_streamed_and_content_sensitive(tmp_path: Path) -> None:
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"weights-v1")
    first = sha256_file(artifact)
    artifact.write_bytes(b"weights-v2")
    assert sha256_file(artifact) != first


def test_directory_hash_covers_relative_names_and_contents(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    nested = tmp_path / "processor"
    nested.mkdir()
    weights = nested / "state.safetensors"
    weights.write_bytes(b"normalizer")
    first = sha256_directory(tmp_path)
    assert sha256_directory(tmp_path) == first
    weights.rename(nested / "renamed.safetensors")
    assert sha256_directory(tmp_path) != first


def test_hash_helpers_reject_missing_or_empty_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing")
    with pytest.raises(ValueError, match="contains no files"):
        sha256_directory(tmp_path)
