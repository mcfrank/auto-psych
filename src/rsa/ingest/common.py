"""The column contract shared by the external sources, and writing one source.

``COLUMNS`` is the pragmods trial table's columns (`src.rsa.pragmods_ingest`)
with ``source`` first and two more at the end:

* ``messages`` — JSON list of the feature indices the speaker could name on
  that trial (`src.rsa.context.Context.messages`); the heard ``utterance``
  is always one of them and true of some object;
* ``covariates`` — JSON object of per-row extras a model may use (e.g. a
  participant's annotated strategy), ``{}`` when there are none.

Participant ids are ``<source>:...`` so that ids never collide across sources,
and experiment names carry a source prefix for the same reason.
"""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from pyprojroot import here

from src.rsa import pragmods_ingest
from src.rsa.context import Context
from src.rsa.dataset import NO_DISPLAY, context_from_row
from src.rsa.ingest.fetch import EXTERNAL_DIR, SourceFile, fetch, sha256_file

REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects" / "rsa_reference" / "data"

COLUMNS: List[str] = ["source", *pragmods_ingest.COLUMNS, "messages", "covariates"]
LISTENER_QUERIES = ("utterance", "prior")


def _json(value) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, sort_keys=isinstance(value, dict))


@dataclass
class Trial:
    """One response, in the order of ``COLUMNS``."""

    source: str
    participant_id: str
    batch: str
    source_file: str
    series: str
    experiment: str
    condition: str
    trial_index: int
    item: str
    feature_names: Sequence[str]
    objects: Sequence[Sequence[int]]
    object_roles: Sequence[str]
    display_order: Sequence[Optional[int]]
    query: str
    query_detail: str
    framing: str
    dv: str
    messages: Sequence[int]
    included: bool
    exclusion_reason: str = ""
    utterance: Optional[int] = None
    choice: Optional[int] = None
    response: object = None
    referent: Optional[int] = None
    paper_cond: str = ""
    notes: List[str] = field(default_factory=list)
    covariates: Dict[str, object] = field(default_factory=dict)

    def as_row(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "participant_id": self.participant_id,
            "batch": self.batch,
            "source_file": self.source_file,
            "series": self.series,
            "experiment": self.experiment,
            "condition": self.condition,
            "paper_cond": self.paper_cond,
            "trial_index": self.trial_index,
            "item": self.item,
            "feature_names": _json(list(self.feature_names)),
            "objects": _json([list(o) for o in self.objects]),
            "object_roles": _json(list(self.object_roles)),
            "display_order": _json(list(self.display_order)),
            "query": self.query,
            "query_detail": self.query_detail,
            "utterance": "" if self.utterance is None else self.utterance,
            "framing": self.framing,
            "familiarization": "",
            "grayscale": "",
            "dv": self.dv,
            "choice": "" if self.choice is None else self.choice,
            "response": "" if self.response is None else _json(self.response),
            "referent": "" if self.referent is None else self.referent,
            "included": self.included,
            "exclusion_reason": self.exclusion_reason,
            "notes": ";".join(self.notes),
            "messages": _json(list(self.messages)),
            "covariates": _json(self.covariates),
        }


@dataclass(frozen=True)
class Source:
    """A pinned external dataset and how to derive its trial table."""

    name: str
    citation: str
    licence: str
    landing_url: str
    files: Tuple[SourceFile, ...]
    build: Callable[[Dict[str, Path]], List[Trial]]
    commit_csv: bool  # only for a licence that permits redistribution (CC-BY)
    pinned_commit: Optional[str] = None

    @property
    def csv_name(self) -> str:
        return f"{self.name}_trials.csv"

    def csv_path(self, cache_dir: Path = EXTERNAL_DIR) -> Path:
        return DATA_DIR / self.csv_name if self.commit_csv else Path(cache_dir) / self.name / self.csv_name

    @property
    def provenance_path(self) -> Path:
        return DATA_DIR / f"{self.name}_trials.provenance.json"


def _repo_relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def frame_of(trials: Sequence[Trial]) -> pd.DataFrame:
    return pd.DataFrame([t.as_row() for t in trials], columns=COLUMNS)


def check_frame(frame: pd.DataFrame, source: str) -> None:
    """Raise unless ``frame`` (as read back from CSV) meets the column contract.

    Every forced-choice row must load as a `Context` (so its utterance is a
    nameable feature true of some object) with its choice an object index.
    """
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"{source}: columns {list(frame.columns)} are not the contract {COLUMNS}")
    if not (frame["source"] == source).all():
        raise ValueError(f"{source}: rows from other sources: {sorted(set(frame['source']) - {source})}")
    bad_ids = frame.loc[~frame["participant_id"].astype(str).str.startswith(f"{source}:"), "participant_id"]
    if len(bad_ids):
        raise ValueError(f"{source}: participant ids without the source prefix, e.g. {bad_ids.iloc[0]!r}")
    dvs = set(frame["dv"])
    if not dvs <= {"forced_choice", "production"}:
        raise ValueError(f"{source}: unexpected dv {sorted(dvs)}")
    production = frame[frame["dv"] == "production"]
    if not (production["query"] == "production").all():
        raise ValueError(f"{source}: production rows must have query=production")
    listener = frame[frame["dv"] == "forced_choice"]
    if not listener["query"].isin(LISTENER_QUERIES).all():
        raise ValueError(f"{source}: forced-choice rows must have query in {LISTENER_QUERIES}")
    if (listener["objects"] == NO_DISPLAY).any():
        raise ValueError(f"{source}: forced-choice rows without a display")
    for i, row in listener.iterrows():
        ctx: Context = context_from_row(row)
        if ctx.messages is None:
            raise ValueError(f"{source} row {i}: no message set")
        choice = int(row["choice"])
        if not 0 <= choice < len(ctx.objects):
            raise ValueError(f"{source} row {i}: choice {choice} is not an object index")


def ingest(source: Source, cache_dir: Path = EXTERNAL_DIR, offline: bool = False, command: str = "") -> Tuple[Path, Path]:
    """Fetch, derive, check and write one source's CSV and provenance."""
    paths = {f.name: fetch(f, cache_dir, source.name, offline=offline) for f in source.files}
    trials = source.build(paths)
    frame = frame_of(trials)
    csv_path = source.csv_path(cache_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    check_frame(pd.read_csv(tmp), source.name)
    tmp.replace(csv_path)
    included = frame[frame["included"]]
    provenance = {
        "source": source.name,
        "citation": source.citation,
        "licence": source.licence,
        "landing_url": source.landing_url,
        "pinned_commit": source.pinned_commit,
        "generation_command": command,
        "source_files": [
            {"name": f.name, "url": f.url, "bytes": f.bytes, "sha256": f.sha256} for f in source.files
        ],
        "output": {
            "path": _repo_relative(csv_path) if source.commit_csv else f"<cache>/{source.name}/{source.csv_name}",
            "committed": source.commit_csv,
            "rows": len(frame),
            "included_rows": len(included),
            "participants": int(frame["participant_id"].nunique()),
            "included_participants": int(included["participant_id"].nunique()),
            "rows_by_experiment": {k: int(v) for k, v in frame.groupby("experiment").size().items()},
            "included_participants_by_experiment": {
                k: int(v) for k, v in included.groupby("experiment")["participant_id"].nunique().items()
            },
            "sha256": sha256_file(csv_path),
        },
    }
    source.provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return csv_path, source.provenance_path


def command_line(module: str) -> str:
    return f"uv run python -m {module} " + " ".join(shlex.quote(a) for a in sys.argv[1:])
