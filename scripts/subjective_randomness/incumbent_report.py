"""CLI: the incumbent record over a finished holdout sweep.

For every cell ``run<r>/<gt>/`` of a sweep, read the inner-loop
``history.json`` of each experiment (from the cell's archived
``agent_runs.tar.gz`` or its kept repo copy), and report the loop-improvement
plan's primary metric: how many scoring steps exported a different best model
than the previous step (incumbent changes), and at how many steps the
incumbent was a model the loop discovered rather than one the cell was seeded
with. The archived ``motif_stack`` cells of ``sweep_rerun`` are the baseline:
0 changes and 0 discovered-incumbent steps over 27 steps.

Reads only; nothing is written into the sweep. The markdown table goes to
``--out`` (and stdout, so a Slurm log carries the numbers) and the per-cell
records as JSON beside it.

Usage:
    uv run python scripts/subjective_randomness/incumbent_report.py \\
        --sweep $SCRATCH/auto-psych/consolidation_2026_09/sweep_rerun \\
        --out   $SCRATCH/auto-psych/consolidation_2026_09/incumbent_sweep_rerun.md \\
        [--gt-model motif_stack]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.incumbent import (  # noqa: E402
    cell_histories,
    incumbent_summary_for_histories,
)


@dataclass
class Args:
    """Report incumbent changes per cell over a finished holdout sweep."""

    sweep: Path
    """Sweep root holding run<r>/<gt>/ cells (each with agent_runs.tar.gz or a
    kept repo copy)."""
    out: Path
    """Markdown report path. The per-cell records are written as JSON beside
    it (same stem, .json)."""
    gt_model: Optional[str] = None
    """Only report the cells of this ground truth."""


def discover_cells(sweep: Path, *, gt_model: Optional[str] = None) -> Dict[str, Path]:
    """``'run<r>/<gt>' -> cell dir`` for every cell directory of the sweep,
    sorted by label. Raises when the root is missing or the selection is empty."""
    sweep = Path(sweep)
    if not sweep.is_dir():
        raise FileNotFoundError(f"The sweep root does not exist: {sweep}")
    cells: Dict[str, Path] = {}
    for run_dir in sorted(sweep.glob("run*")):
        if not run_dir.is_dir():
            continue
        for cell_dir in sorted(run_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            if gt_model is not None and cell_dir.name != gt_model:
                continue
            cells[f"{run_dir.name}/{cell_dir.name}"] = cell_dir
    if not cells:
        selection = f" for ground truth {gt_model!r}" if gt_model else ""
        raise FileNotFoundError(f"No run<r>/<gt>/ cells{selection} under {sweep}")
    return cells


def incumbent_records(cells: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    """The per-cell incumbent summary for every cell, in label order."""
    records: Dict[str, Dict[str, Any]] = {}
    for label, cell_dir in cells.items():
        try:
            histories = cell_histories(cell_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise type(exc)(f"{label}: {exc}") from exc
        records[label] = incumbent_summary_for_histories(histories)
    return records


def totals(records: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    return {
        "n_cells": len(records),
        "n_steps": sum(r["n_steps"] for r in records.values()),
        "n_incumbent_changes": sum(r["n_incumbent_changes"] for r in records.values()),
        "n_steps_discovered_incumbent": sum(
            r["n_steps_discovered_incumbent"] for r in records.values()
        ),
        "n_cells_with_a_change": sum(
            1 for r in records.values() if r["n_incumbent_changes"] > 0
        ),
        "n_cells_with_a_discovered_incumbent": sum(
            1 for r in records.values() if r["n_steps_discovered_incumbent"] > 0
        ),
    }


def _format_changes(record: Dict[str, Any]) -> str:
    if not record["changes"]:
        return "none"
    return "; ".join(
        f"{c['from']} -> {c['to']} (experiment {c['experiment']}, step {c['step']})"
        for c in record["changes"]
    )


def render_markdown(sweep: Path, records: Dict[str, Dict[str, Any]]) -> str:
    total = totals(records)
    lines = [
        f"# Incumbent record: `{sweep}`",
        "",
        "Per cell: scoring steps, steps at which the exported best model differed "
        "from the previous step's, steps at which it was a discovered model (not "
        "scored at experiment 1's seed step), the final incumbent, and the changes.",
        "",
        "| cell | steps | incumbent changes | discovered-incumbent steps | final incumbent | changes |",
        "|---|---|---|---|---|---|",
    ]
    for label, record in records.items():
        lines.append(
            f"| {label} | {record['n_steps']} | {record['n_incumbent_changes']} | "
            f"{record['n_steps_discovered_incumbent']} | {record['final_incumbent']} | "
            f"{_format_changes(record)} |"
        )
    lines += [
        "",
        f"**Total:** {total['n_cells']} cells, {total['n_steps']} steps, "
        f"{total['n_incumbent_changes']} incumbent changes, "
        f"{total['n_steps_discovered_incumbent']} steps with a discovered incumbent; "
        f"{total['n_cells_with_a_change']} of {total['n_cells']} cells changed incumbent "
        f"at least once and {total['n_cells_with_a_discovered_incumbent']} ever had a "
        "discovered incumbent.",
        "",
    ]
    return "\n".join(lines)


def main(args: Args) -> None:
    cells = discover_cells(args.sweep, gt_model=args.gt_model)
    records = incumbent_records(cells)
    markdown = render_markdown(args.sweep, records)
    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")
    out_json = out_md.with_suffix(".json")
    out_json.write_text(
        json.dumps(
            {"sweep": str(args.sweep), "cells": records, "totals": totals(records)},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(markdown)
    print(f"Wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
