"""Concatenate chosen sources into one responses CSV for the loop.

    uv run python -m src.rsa.ingest.combine \\
        --sources pragmods mayn_demberg_2026 sikos_2021 --out <path>

Reads each source's derived CSV (pragmods' committed table, a CC-BY source's
committed CSV, or a cached CSV built by ``src.rsa.ingest.run``; a missing one
raises with the command that builds it), checks that it meets the column
contract (`src.rsa.ingest.common.COLUMNS`), and writes them in the given
order. The pragmods table predates ``source``, ``messages`` and
``covariates``: it is given ``source=pragmods``, empty ``messages`` (every
feature is a word) and ``{}``, and its participant ids the ``pragmods:``
prefix. Experiment names and participant ids must not collide across sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import pandas as pd
import tyro

from src.rsa import pragmods_ingest
from src.rsa.dataset import DEFAULT_TRIALS_CSV
from src.rsa.ingest.common import COLUMNS
from src.rsa.ingest.fetch import EXTERNAL_DIR
from src.rsa.ingest.run import SOURCES

PRAGMODS = "pragmods"
CombineSource = Literal["pragmods", "mayn_demberg_2026", "mayn_demberg_2023", "mayn_demberg_2022", "sikos_2021"]


def load_pragmods(path: Path = DEFAULT_TRIALS_CSV) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != pragmods_ingest.COLUMNS:
        raise ValueError(f"{path}: columns are not the pragmods contract {pragmods_ingest.COLUMNS}")
    frame.insert(0, "source", PRAGMODS)
    frame["participant_id"] = PRAGMODS + ":" + frame["participant_id"]
    frame["messages"] = ""
    frame["covariates"] = "{}"
    return frame[COLUMNS]


def load_source(name: str, cache_dir: Path = EXTERNAL_DIR) -> pd.DataFrame:
    if name == PRAGMODS:
        return load_pragmods()
    if name not in SOURCES:
        raise ValueError(f"unknown source {name!r}; known: {[PRAGMODS, *sorted(SOURCES)]}")
    path = SOURCES[name].csv_path(cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; build it with: uv run python -m src.rsa.ingest.run --sources {name}"
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"{path}: columns {list(frame.columns)} are not the contract {COLUMNS}")
    return frame


def combine(sources: List[str], cache_dir: Path = EXTERNAL_DIR) -> pd.DataFrame:
    if len(set(sources)) != len(sources):
        raise ValueError(f"a source is listed twice: {sources}")
    frames = [load_source(name, cache_dir) for name in sources]
    owner = {}
    for name, frame in zip(sources, frames):
        if not (frame["source"] == name).all():
            raise ValueError(f"{name}: its CSV has rows of other sources")
        if not frame["participant_id"].str.startswith(f"{name}:").all():
            raise ValueError(f"{name}: participant ids without the {name}: prefix")
        for experiment in frame["experiment"].unique():
            if experiment in owner:
                raise ValueError(f"experiment {experiment!r} is in both {owner[experiment]} and {name}")
            owner[experiment] = name
    return pd.concat(frames, ignore_index=True)[COLUMNS]


@dataclass
class Args:
    """Combine derived trial CSVs (pragmods and external sources) into one."""

    sources: List[CombineSource]
    """Sources, in output order."""
    out: Path
    """Output CSV path."""
    cache_dir: Path = EXTERNAL_DIR
    """Where the uncommitted sources' CSVs were built (src.rsa.ingest.run --cache-dir)."""


def main(args: Args) -> None:
    combined = combine(list(args.sources), args.cache_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    combined.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(out)
    counts = combined.assign(included=combined["included"] == "True").groupby("source")["included"].agg(["size", "sum"])
    print(f"wrote {out}: {len(combined)} rows")
    print(counts.rename(columns={"size": "rows", "sum": "included_rows"}).to_string())


if __name__ == "__main__":
    main(tyro.cli(Args))
