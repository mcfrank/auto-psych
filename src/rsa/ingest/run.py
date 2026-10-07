"""Fetch, derive and write external reference-game datasets.

    uv run python -m src.rsa.ingest.run --sources mayn_demberg_2026 sikos_2021 \\
        mayn_demberg_2023 mayn_demberg_2022

For each source: download its pinned files into ``--cache-dir`` (verified by
sha256), derive the canonical trial CSV, check it against the column contract
and write it with a ``<source>_trials.provenance.json`` pin file under the
project's data directory. A CC-BY source's CSV is written there too (and is
committed); a source without a licence is written to
``<cache-dir>/<source>/<source>_trials.csv`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

import tyro

from src.rsa.ingest.common import Source, command_line, ingest
from src.rsa.ingest.fetch import EXTERNAL_DIR
from src.rsa.ingest.mayn_demberg import MAYN_DEMBERG_2022, MAYN_DEMBERG_2023, MAYN_DEMBERG_2026
from src.rsa.ingest.sikos2021 import SIKOS_2021

SOURCES: Dict[str, Source] = {
    s.name: s for s in (MAYN_DEMBERG_2026, MAYN_DEMBERG_2023, MAYN_DEMBERG_2022, SIKOS_2021)
}

# Considered and deliberately not ingested (README "Not ingested").
NOT_INGESTED: Dict[str, str] = {
    "duff_mayn_demberg_2026": (
        "Duff, Mayn & Demberg (2026, Open Mind; OSF 7uwx9/ad685) gave participants feedback after every "
        "reference-game trial ('indicating whether their response was the intended target') and a speaker "
        "pre-training, so its listener choices are learned under reinforcement, not one-shot interpretations."
    ),
}

SourceName = Literal["mayn_demberg_2026", "mayn_demberg_2023", "mayn_demberg_2022", "sikos_2021"]


@dataclass
class Args:
    """Build the canonical trial CSVs of external reference-game datasets."""

    sources: List[SourceName]
    """Sources to fetch and derive."""
    cache_dir: Path = EXTERNAL_DIR
    """Download cache (and output directory of sources without a licence); gitignored."""
    offline: bool = False
    """Use only already-cached downloads (raise when one is missing)."""


def main(args: Args) -> None:
    unknown = sorted(set(args.sources) - set(SOURCES))
    if unknown:
        raise ValueError(f"unknown sources {unknown}; known: {sorted(SOURCES)}")
    command = command_line("src.rsa.ingest.run")
    for name in args.sources:
        csv_path, prov_path = ingest(SOURCES[name], args.cache_dir, args.offline, command)
        print(f"{name}: wrote {csv_path}")
        print(f"{name}: wrote {prov_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
