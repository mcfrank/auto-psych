"""Forced-choice trials of the canonical pragmods table as contexts and choices.

Reads `projects/rsa_reference/data/pragmods_trials.csv` (built by
`src.rsa.pragmods_ingest`). Only included forced-choice rows with a recorded
display are trials; the speakers' follow-up listener trials have no recorded
matrix and are left out by name (`NO_DISPLAY`), never silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.rsa.context import Context
from src.runtime.config import PROJECT_ASSETS_DIR

DEFAULT_TRIALS_CSV = PROJECT_ASSETS_DIR / "rsa_reference" / "data" / "pragmods_trials.csv"
FAMILIARIZATION_IMAGES = 9

# framing label -> valence of the speaker's attitude to the object he means.
VALENCE_BY_FRAMING = {
    "my_favorite_X_has": 1,
    "silent_favorite": 1,
    "my_least_favorite_X_has": -1,
    "silent_least_favorite": -1,
    "one_word": 0,
    "my_X_has": 0,
    "the_X_has": 0,
    "tricky_guy": 0,
    "points_to_color_patch": 0,
    # The message is a picture of one feature (Mayn & Demberg's games).
    "message_icon": 0,
}

NO_DISPLAY = "[]"


@dataclass
class Trials:
    contexts: List[Context]
    choices: List[int]
    frame: pd.DataFrame  # the CSV rows, aligned with contexts


def _json_or_none(value) -> Optional[list]:
    if isinstance(value, float) and pd.isna(value):
        return None
    if value is None or value == "":
        return None
    return json.loads(value)


def context_from_row(row: pd.Series) -> Context:
    framing = row["framing"]
    if framing not in VALENCE_BY_FRAMING:
        raise ValueError(f"framing {framing!r} has no valence in VALENCE_BY_FRAMING")
    objects = json.loads(row["objects"])
    names = json.loads(row["feature_names"])
    names = tuple(n if n is not None else f"feature{i}" for i, n in enumerate(names))
    if row["query"] == "utterance":
        utterance: Optional[int] = int(row["utterance"])
    elif row["query"] == "prior":
        utterance = None
    else:
        raise ValueError(f"query {row['query']!r} is not a listener trial")
    fam = _json_or_none(row["familiarization"])
    if fam is not None:
        if sum(fam) != FAMILIARIZATION_IMAGES:
            raise ValueError(f"familiarization counts {fam} do not sum to {FAMILIARIZATION_IMAGES}")
        fam = tuple(c / FAMILIARIZATION_IMAGES for c in fam)
    gray = _json_or_none(row["grayscale"])
    # Optional column: the feature indices the speaker could name (a dataset
    # whose every feature is a word, like pragmods, omits it).
    messages = _json_or_none(row["messages"]) if "messages" in row.index else None
    return Context(
        objects=tuple(tuple(r) for r in objects),
        feature_names=names,
        utterance=utterance,
        item=str(row["item"]),
        familiarization=fam,
        grayscale=None if gray is None else tuple(gray),
        valence=VALENCE_BY_FRAMING[framing],
        messages=None if messages is None else tuple(messages),
    )


def load_forced_choice(
    path: Path = DEFAULT_TRIALS_CSV, experiments: Optional[Sequence[str]] = None
) -> Trials:
    """Included forced-choice listener trials, optionally from some experiments only."""
    df = pd.read_csv(path)
    rows = df[(df["dv"] == "forced_choice") & df["included"].astype(bool)]
    rows = rows[rows["objects"] != NO_DISPLAY]
    if experiments is not None:
        unknown = sorted(set(experiments) - set(df["experiment"]))
        if unknown:
            raise ValueError(f"no experiments named {unknown} in {path}")
        rows = rows[rows["experiment"].isin(experiments)]
    rows = rows.reset_index(drop=True)
    contexts = [context_from_row(r) for _, r in rows.iterrows()]
    choices = [int(c) for c in rows["choice"]]
    return Trials(contexts=contexts, choices=choices, frame=rows)


def is_plain(ctx: Context) -> bool:
    """A display the live experiments can show: no valence framing, no
    familiarization, no greyscale (PI 2026-10-10: the live phase's scope)."""
    return ctx.valence in (0, None) and ctx.familiarization is None and ctx.grayscale is None


def write_plain_trials(path: Path, out: Path) -> dict:
    """The included forced-choice trials of ``path`` on plain displays, as a
    trials CSV at ``out``; returns what was left out, by source and experiment."""
    trials = load_forced_choice(path)
    keep = np.array([is_plain(c) for c in trials.contexts], dtype=bool)
    dropped = trials.frame[~keep]
    src = dropped["source"] if "source" in dropped.columns else pd.Series("pragmods", index=dropped.index)
    left_out = {f"{a}|{b}": int(n) for (a, b), n in dropped.groupby([src, dropped["experiment"]]).size().items()}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.csv")
    trials.frame[keep].to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(out)
    return dict(n_kept=int(keep.sum()), n_left_out=int((~keep).sum()), left_out=left_out)

