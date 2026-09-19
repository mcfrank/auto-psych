"""Ledger of every hypothesis the inner loop has tried (``attempted_hypotheses.jsonl``).

The zoo (``models/`` + its manifest) only ever shows the *current* model set.
A candidate that was rejected at admission, or a model that was pruned after
losing, disappears from ``existing_hypotheses.md`` at once — and the next
round's candidate agents (or the next experiment's) propose it again under
a new name, wasting candidate slots.

The ledger is the loop's memory: one JSON line per event, appended the moment
it happens so a crashed run still leaves the record, started from the ledger
the previous experiment carried in ``cognitive_models/``, and rendered into
every candidate brief as the "already tried — do not re-propose" section
(``render_markdown``).  See ``docs/consolidation_decision_record.md`` §
"History of specific choices" for the empirical evidence that motivated this
module. Events:

- ``admitted`` — the candidate entered the zoo (its later fate, if any, is a
  later line under the same name);
- ``rejected`` — the candidate never entered (no files, unloadable, unfittable,
  non-finite ELPD, or a near-duplicate of an admitted model);
- ``pruned`` — an admitted model lost to the best by more than the pruning
  margin and moved to ``models/pruned/``;
- ``dropped`` — a seeded/carried model could not be fit or scored on this
  experiment's data.

Every line carries the model name, a one-line hypothesis, a human-readable
``detail`` (the margin, the nearest neighbour, the error) and a ``context``
string locating the event (experiment, round, candidate slot, lens).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, List, Optional

LEDGER_FILENAME = "attempted_hypotheses.jsonl"
OUTCOMES = ("admitted", "rejected", "pruned", "dropped")

# Hypotheses are 1–3 sentences; the brief shows one line per retired model.
ONE_LINE_LIMIT = 240


def one_line(text: str, *, limit: int = ONE_LINE_LIMIT) -> str:
    """Collapse ``text`` to a single line of at most ``limit`` characters."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


@dataclass(frozen=True)
class LedgerEntry:
    """One event in the hypothesis ledger: a candidate admitted, rejected, pruned, or dropped."""

    name: str
    outcome: str
    detail: str
    hypothesis: str
    context: str

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(
                f"Ledger outcome must be one of {OUTCOMES}; got {self.outcome!r} "
                f"for model {self.name!r}."
            )

    def to_json(self) -> str:
        """Serialize to a single JSON line for appending to the JSONL ledger."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str, *, source: Path) -> "LedgerEntry":
        """Deserialize one JSON line. Raises ``ValueError`` on malformed input."""
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed line in the hypothesis ledger {source}: {exc}: {line!r}"
            ) from exc
        expected = {f.name for f in fields(cls)}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError(
                f"Ledger line in {source} must be an object with exactly the keys "
                f"{sorted(expected)}; got {line!r}"
            )
        return cls(**data)


class HypothesisLedger:
    """Append-only JSONL ledger at ``path`` (see the module docstring)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @classmethod
    def create(
        cls, path: Path, *, inherit_from: Optional[Path]
    ) -> "HypothesisLedger":
        """Start a fresh ledger at ``path``, seeded from ``inherit_from`` if it exists.

        ``path`` is overwritten: the loop that owns it re-seeds its zoo on every
        start, so a ledger left by an earlier run into the same results dir
        would double-count. An inherited file that fails to parse raises here,
        before any work is done on top of it.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if inherit_from is not None and Path(inherit_from).exists():
            shutil.copyfile(inherit_from, path)
            ledger = cls(path)
            ledger.entries()
            return ledger
        path.write_text("", encoding="utf-8")
        return cls(path)

    def append(self, entry: LedgerEntry) -> None:
        """Append one event to the ledger file (one JSON line per call)."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(entry.to_json() + "\n")

    def entries(self) -> List[LedgerEntry]:
        """All entries in chronological order. Raises ``FileNotFoundError`` if the ledger is missing."""
        if not self.path.exists():
            raise FileNotFoundError(f"Hypothesis ledger does not exist: {self.path}")
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [
            LedgerEntry.from_json(line, source=self.path)
            for line in lines
            if line.strip()
        ]

    def retired(self, live_names: Iterable[str]) -> List[LedgerEntry]:
        """The latest entry per model name that is NOT in ``live_names``.

        Ordered by each name's first appearance in the ledger. A live model's
        history (admitted, perhaps later pruned and re-admitted under the same
        name) is the zoo's business; the brief shows the live set separately.
        """
        live = set(live_names)
        latest: dict = {}
        for entry in self.entries():
            latest[entry.name] = entry
        return [entry for name, entry in latest.items() if name not in live]

    def render_markdown(self, live_names: Iterable[str]) -> str:
        """The candidate brief's "already tried — do not re-propose" section."""
        retired = self.retired(live_names)
        header = "# Already tried — do not re-propose\n\n"
        if not retired:
            return header + (
                "No earlier hypothesis has been retired yet: every hypothesis "
                "tried so far is still in the model set (see "
                "`existing_hypotheses.md`).\n"
            )
        lines = [
            header
            + f"{len(retired)} hypotheses proposed earlier in this project are no "
            "longer in the model set. Do not propose any of them again, under any "
            "name: a genuinely new hypothesis differs in *mechanism*, not in "
            "parameterisation or wording. A *pruned* row is a mechanism the data "
            "ruled out, with its margin behind the model that beat it; a "
            "*rejected* row is a candidate that never entered the set — most often "
            "a near-duplicate of a model still in the set, meaning that region of "
            "hypothesis space is already covered. Pruned models stay readable under "
            "`models/pruned/`.\n",
            "| model | outcome | hypothesis |",
            "| --- | --- | --- |",
        ]
        for entry in retired:
            outcome = f"{entry.outcome} ({entry.context})" if entry.context else entry.outcome
            if entry.detail:
                outcome += f": {entry.detail}"
            lines.append(
                f"| {entry.name} | {_cell(outcome)} | {_cell(entry.hypothesis)} |"
            )
        return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    """One markdown table cell: single line, no pipes."""
    return one_line(text).replace("|", "/")
