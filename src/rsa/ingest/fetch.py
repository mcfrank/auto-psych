"""Pinned downloads: fetch a public source file once and verify its sha256.

A file is cached at ``<cache>/<source>/raw/<name>``. A cached file is
re-verified on every use and never silently re-downloaded: a hash mismatch,
cached or fresh, raises (``PinMismatch``) and names the file to delete. OSF
files are fetched from ``https://osf.io/download/<id>/``, GitHub files from
``raw.githubusercontent.com`` at a pinned commit, PLoS supporting files from
the journal's ``article/file?type=supplementary`` links.

Downloads are untrusted data: they are only ever read as CSV text by the
source modules, never executed or unpickled.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from pyprojroot import here

EXTERNAL_DIR = here() / "data" / "rsa" / "external"
DOWNLOAD_TIMEOUT_SEC = 300
_CHUNK = 1 << 20


class PinMismatch(RuntimeError):
    """A downloaded or cached file does not have its pinned sha256."""


@dataclass(frozen=True)
class SourceFile:
    """One pinned public file of a source."""

    name: str  # path under the source's raw cache, e.g. "data/main_task_data.csv"
    url: str
    sha256: str
    bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_dir(cache_dir: Path, source: str) -> Path:
    return Path(cache_dir) / source / "raw"


def cached_path(cache_dir: Path, source: str, file: SourceFile) -> Path:
    return raw_dir(cache_dir, source) / file.name


def verify(path: Path, file: SourceFile) -> None:
    got = sha256_file(path)
    if got != file.sha256:
        raise PinMismatch(
            f"{path} has sha256 {got}, but {file.url} is pinned at {file.sha256}. "
            f"The source changed or the file is corrupt; delete {path} to re-fetch, "
            f"and update the pin only after checking what changed."
        )


def fetch(file: SourceFile, cache_dir: Path, source: str, *, offline: bool = False) -> Path:
    """The verified local copy of ``file``, downloading it when not cached."""
    path = cached_path(cache_dir, source, file)
    if path.exists():
        verify(path, file)
        return path
    if offline:
        raise FileNotFoundError(f"{path} is not cached and offline=True (source {file.url})")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".part{os.getpid()}")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(file.url, timeout=DOWNLOAD_TIMEOUT_SEC) as response, open(tmp, "wb") as out:
            for chunk in iter(lambda: response.read(_CHUNK), b""):
                digest.update(chunk)
                out.write(chunk)
        got = digest.hexdigest()
        if got != file.sha256:
            raise PinMismatch(
                f"{file.url} downloaded with sha256 {got}, pinned {file.sha256} "
                f"({tmp.stat().st_size} bytes, pinned {file.bytes}). Not using it."
            )
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path
