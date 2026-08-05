"""Stable, streaming identities for local experiment artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


_HASH_CHUNK_BYTES = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Hash a file without loading large checkpoints into host memory."""

    if not path.is_file():
        raise FileNotFoundError(f"Artifact file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(root: Path) -> str:
    """Hash relative paths and contents for every file below a directory."""

    if not root.is_dir():
        raise FileNotFoundError(f"Artifact directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Artifact directory contains no files: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
    return digest.hexdigest()
