"""Writes that a crash leaves either undone or done, never half done.

A resumed run decides what to redo from what is on disk, so a file or
directory that a killed process left half written reads as finished work: a
truncated ``responses.csv`` that still has a header and one row passes the
collect validator, and a model set whose manifest was copied before its ledger
passes the model-set validator with its ledger silently gone. Everything here
builds the new content beside its destination and renames it into place, and
a rename within one filesystem is atomic.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable


def write_text_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file in the same directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def replace_directory(dest: Path, build: Callable[[Path], None]) -> None:
    """Make ``dest`` hold exactly what ``build`` writes, or leave it as it was.

    ``build`` fills a fresh temporary directory beside ``dest``; only when it
    returns is the old ``dest`` removed and the temporary directory renamed to
    it. A crash inside ``build`` leaves ``dest`` untouched (the leftover
    temporary directory is cleared by the next call); a crash between the
    removal and the rename leaves no ``dest``, which every caller reads as
    "not done" and redoes.
    """
    dest = Path(dest)
    staging = dest.with_name(f".{dest.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    build(staging)
    if dest.exists():
        shutil.rmtree(dest)
    os.rename(staging, dest)


def copy_directory_atomically(src: Path, dest: Path) -> None:
    """Replace ``dest`` with a copy of ``src`` (``replace_directory``)."""

    def build(staging: Path) -> None:
        shutil.copytree(src, staging, dirs_exist_ok=True)

    replace_directory(dest, build)
