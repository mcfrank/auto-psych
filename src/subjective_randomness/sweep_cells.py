"""Which cells of a holdout sweep a summary covers, and which it does not.

Every sweep summary used to read whatever ``run<r>/<gt>/`` cells happened to
have a result: a cell that died on a time limit or ran out of memory simply
vanished from the report (first audit C6), and the incumbent report read the
kept run records of cells that never finished as if they were whole. Failures
plausibly correlate with ground-truth difficulty, so both bias the means.

``survey_sweep`` sorts the sweep's expected cells into

* **complete** — ``holdout.json`` exists (the array task writes it last, after
  the whole cell was scored);
* **partial** — the cell directory exists but has no ``holdout.json``: the
  cell started and did not finish (still running, or failed and not yet
  resumed);
* **missing** — no cell directory at all: the task never started.

The expected cells are ``run1..run<n_repeats>`` × ``gt_models`` when the
caller states them (the sweep's ``N_REPEATS``/``GT_MODELS``). Otherwise they
are inferred from what is on disk — ``run1..run<highest r present>`` × every
ground-truth directory present in any repeat — and the survey says so: a
repeat or ground truth with no directory anywhere cannot be seen that way.

``accounting_lines`` renders the survey as markdown for a report, so every
summary lists what it includes and what it leaves out, and why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

RESULT_FILENAME = "holdout.json"

_RUN_DIR_RE = re.compile(r"run(\d+)")


@dataclass
class SweepSurvey:
    """The expected cells of a sweep, sorted by what they hold."""

    root: Path
    complete: Dict[str, Path] = field(default_factory=dict)
    """``'run<r>/<gt>' -> cell dir`` for every cell with a ``holdout.json``."""
    partial: Dict[str, str] = field(default_factory=dict)
    """``'run<r>/<gt>' -> reason`` for every started cell without one."""
    missing: Dict[str, str] = field(default_factory=dict)
    """``'run<r>/<gt>' -> reason`` for every expected cell with no directory."""
    expected_from: str = ""
    """How the expected cells were determined (stated, or inferred from disk)."""

    @property
    def n_expected(self) -> int:
        return len(self.complete) + len(self.partial) + len(self.missing)

    def select(self, gt_model: Optional[str]) -> "SweepSurvey":
        """The same survey restricted to one ground truth (``None``: all)."""
        if gt_model is None:
            return self

        def keep(label: str) -> bool:
            return label.split("/", 1)[1] == gt_model

        return SweepSurvey(
            root=self.root,
            complete={k: v for k, v in self.complete.items() if keep(k)},
            partial={k: v for k, v in self.partial.items() if keep(k)},
            missing={k: v for k, v in self.missing.items() if keep(k)},
            expected_from=f"{self.expected_from}; restricted to {gt_model}",
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "root": str(self.root),
            "expected_from": self.expected_from,
            "n_expected": self.n_expected,
            "complete": sorted(self.complete),
            "partial": dict(sorted(self.partial.items())),
            "missing": dict(sorted(self.missing.items())),
        }


def _repeat_number(run_dir: Path) -> Optional[int]:
    match = _RUN_DIR_RE.fullmatch(run_dir.name)
    return int(match.group(1)) if match else None


def survey_sweep(
    root: Path,
    *,
    n_repeats: Optional[int] = None,
    gt_models: Optional[Sequence[str]] = None,
) -> SweepSurvey:
    """Sort a sweep's expected ``run<r>/<gt>`` cells into complete, partial and
    missing (see the module docstring). ``n_repeats`` and ``gt_models`` state
    the expected grid; each one not given is inferred from the directories
    present. Raises when the root does not exist or nothing is expected."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"The sweep root does not exist: {root}")
    run_dirs = {
        number: path
        for path in root.iterdir()
        if path.is_dir() and (number := _repeat_number(path)) is not None
    }
    stated = []
    inferred = []
    if n_repeats is None:
        n_repeats = max(run_dirs, default=0)
        inferred.append(f"repeats run1..run{n_repeats} (the highest run<r> present)")
    else:
        stated.append(f"{n_repeats} repeats")
    if gt_models is None:
        gt_models = sorted(
            {
                cell.name
                for path in run_dirs.values()
                for cell in path.iterdir()
                if cell.is_dir()
            }
        )
        inferred.append(
            f"ground truths {', '.join(gt_models) or '(none)'} (every one present in some repeat)"
        )
    else:
        stated.append(f"ground truths {', '.join(gt_models)}")
    if n_repeats < 1 or not gt_models:
        raise FileNotFoundError(f"No run<r>/<gt>/ cells expected or found under {root}")

    parts = []
    if stated:
        parts.append("stated: " + "; ".join(stated))
    if inferred:
        parts.append(
            "inferred from the directories present: "
            + "; ".join(inferred)
            + " (a repeat or ground truth with no directory anywhere is not seen; "
            "pass the sweep's repeat count and ground truths to state them)"
        )
    survey = SweepSurvey(root=root, expected_from="; ".join(parts))
    for repeat in range(1, n_repeats + 1):
        for gt in gt_models:
            label = f"run{repeat}/{gt}"
            cell_dir = root / f"run{repeat}" / gt
            if (cell_dir / RESULT_FILENAME).is_file():
                survey.complete[label] = cell_dir
            elif cell_dir.is_dir():
                survey.partial[label] = (
                    f"started but has no {RESULT_FILENAME}: the cell did not finish "
                    "(still running, or failed and not resumed)"
                )
            else:
                survey.missing[label] = "no cell directory: the task never started"
    return survey


def accounting_lines(survey: SweepSurvey, *, included: Sequence[str]) -> List[str]:
    """Markdown lines saying which cells a summary includes and which it does
    not, and why. ``included`` are the cells the summary's numbers cover (a
    subset of the complete ones; a complete cell left out of the numbers is
    listed by the summary itself with its own reason)."""
    lines = [
        "## Cells",
        "",
        f"Expected: {survey.n_expected} ({survey.expected_from}). "
        f"Complete: {len(survey.complete)}. Partial: {len(survey.partial)}. "
        f"Missing: {len(survey.missing)}. Included in the numbers below: {len(included)}.",
        "",
    ]
    if survey.partial or survey.missing:
        lines.append("Left out (no result to summarise):")
        lines.append("")
        for label, reason in sorted({**survey.partial, **survey.missing}.items()):
            status = "partial" if label in survey.partial else "missing"
            lines.append(f"- `{label}` ({status}): {reason}")
        lines.append("")
    return lines
