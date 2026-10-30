"""Atomic writes stay correct when two processes write the same file at once."""

from __future__ import annotations

import os

from src.runtime import atomic_files
from src.runtime.atomic_files import write_text_atomically


def test_concurrent_writers_of_one_file_do_not_collide(tmp_path, monkeypatch):
    # Reproduces the figure-data shards that wrote pooled/responses.csv at the
    # same moment: a second writer finishes between the first writer's write
    # of its temporary file and its rename. With one shared temporary name the
    # second writer renames the first one's file away and the first rename
    # raises FileNotFoundError.
    path = tmp_path / "responses.csv"
    real_replace = os.replace
    interrupted = False

    def replace_after_another_writer(src, dst):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            write_text_atomically(path, "second\n")
        real_replace(src, dst)

    monkeypatch.setattr(atomic_files.os, "replace", replace_after_another_writer)
    write_text_atomically(path, "first\n")

    assert path.read_text() == "first\n"
    assert [p.name for p in tmp_path.iterdir()] == ["responses.csv"]
