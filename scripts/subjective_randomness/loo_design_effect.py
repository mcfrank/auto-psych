"""CLI: the LOO design effect over a finished holdout sweep (no new MCMC).

For every cell ``run<r>/<gt>/`` of a sweep, rebuild each recorded scoring
step's model comparison from the cell's ``mcmc_cache/`` and report it under
three units — the loop's trial-level PSIS-LOO, a stimulus-clustered standard
error of the same ELPD difference, and a leave-one-stimulus-out PSIS-LOO —
with each unit's prune set under the loop's ``elpd_diff > 2·dse`` rule. See
``src/subjective_randomness/loo_design_effect.py`` for the definitions and
the checks that tie every reproduced number to the run record.

Reads only; nothing is written into the sweep. Archive members are extracted
under ``--work-dir``. One JSON record per cell goes to ``--out-dir`` as
``<run>__<gt>.json`` and the sweep table to ``<out-dir>/loo_design_effect.md``
(also printed, so a log carries the numbers). ``--cells`` selects a subset —
the per-cell records are independent, so several invocations can split a
sweep and ``--report-only`` then rebuilds the table from every record present.

A cell that finished the harness has ``holdout.json`` with the sampler kwargs
its fits were cached under; an unfinished cell (kept repo copy, no
``holdout.json``) needs them from ``--fit-kwargs-json``.

Usage:
    uv run python scripts/subjective_randomness/loo_design_effect.py \\
        --sweep    $SCRATCH/auto-psych/consolidation_2026_09/sweep_rerun \\
        --out-dir  $SCRATCH/auto-psych/consolidation_2026_09/loo_design_effect \\
        --work-dir $SCRATCH/auto-psych/consolidation_2026_09/loo_design_effect/work \\
        [--cells run2/motif_stack,run3/motif_stack] [--report-only]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.pipelines.inner_loop.model_zoo import DEFAULT_PRUNE_DSE_MULTIPLIER  # noqa: E402
from src.subjective_randomness.loo_design_effect import analyze_cell  # noqa: E402

TABLE_NAME = "loo_design_effect.md"


@dataclass
class Args:
    """The LOO design effect over a finished holdout sweep."""

    sweep: Path
    """Sweep root holding run<r>/<gt>/ cells, each with mcmc_cache/ and either
    agent_runs.tar.gz or a kept repo copy."""
    out_dir: Path
    """Where the per-cell JSON records and the markdown table are written."""
    work_dir: Path
    """Where archive members are extracted (never into the sweep)."""
    cells: Optional[str] = None
    """Comma-separated cell labels (run<r>/<gt>) to analyse; default all."""
    fit_kwargs_json: Optional[str] = None
    """JSON object of the sampler kwargs the fits were cached under, for cells
    without a holdout.json (must agree with holdout.json where one exists)."""
    dse_multiplier: float = DEFAULT_PRUNE_DSE_MULTIPLIER
    """The loop's pruning threshold multiplier."""
    report_only: bool = False
    """Skip the analysis; rebuild the table from the records already in out_dir."""


def discover_cells(sweep: Path, *, labels: Optional[List[str]] = None) -> Dict[str, Path]:
    """``'run<r>/<gt>' -> cell dir`` for every cell of the sweep that has an
    ``mcmc_cache/``, sorted by label; or exactly the requested labels, raising
    on any that is not such a cell."""
    sweep = Path(sweep)
    if not sweep.is_dir():
        raise FileNotFoundError(f"The sweep root does not exist: {sweep}")
    cells: Dict[str, Path] = {}
    for run_dir in sorted(sweep.glob("run*")):
        if not run_dir.is_dir():
            continue
        for cell_dir in sorted(run_dir.iterdir()):
            if cell_dir.is_dir() and (cell_dir / "mcmc_cache").is_dir():
                cells[f"{run_dir.name}/{cell_dir.name}"] = cell_dir
    if labels is None:
        if not cells:
            raise FileNotFoundError(f"No run<r>/<gt>/ cells with mcmc_cache/ under {sweep}")
        return cells
    unknown = [label for label in labels if label not in cells]
    if unknown:
        raise FileNotFoundError(
            f"Not cells with mcmc_cache/ under {sweep}: {unknown}; known: {sorted(cells)}"
        )
    return {label: cells[label] for label in labels}


def record_path(out_dir: Path, label: str) -> Path:
    return Path(out_dir) / f"{label.replace('/', '__')}.json"


def load_records(out_dir: Path) -> List[Dict[str, Any]]:
    """Every per-cell record in ``out_dir``, in label order."""
    paths = sorted(Path(out_dir).glob("run*__*.json"))
    if not paths:
        raise FileNotFoundError(f"No per-cell records (run*__*.json) under {out_dir}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _quantile_cell(q: Mapping[str, Any]) -> str:
    if not q["n"]:
        return "—"
    return f"{q['median']:.2f} [{q['min']:.2f}, {q['max']:.2f}]"


def render_table(records: List[Mapping[str, Any]]) -> str:
    """The sweep table: one row per cell and a totals row."""
    header = [
        "cell",
        "run record",
        "trials / stimulus by experiment",
        "decisions",
        "pruned by loop / decisions",
        "cluster prunes (flips)",
        "grouped prunes, reliability-gated (flips)",
        "grouped prunes, ungated (flips)",
        "grouped-unreliable decisions",
        "dse ratio cluster/trial, median [min, max]",
        "dse ratio grouped/trial, median [min, max]",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    totals = {
        "n_decisions": 0,
        "n_pruned_archived": 0,
        "n_prune_cluster": 0,
        "n_flip_trial_vs_cluster": 0,
        "n_prune_grouped": 0,
        "n_flip_trial_vs_grouped": 0,
        "n_prune_grouped_ignoring_reliability": 0,
        "n_flip_trial_vs_grouped_ignoring_reliability": 0,
        "n_grouped_unreliable_decisions": 0,
    }
    for record in records:
        s = record["summary"]
        for key in totals:
            totals[key] += int(s[key])
        per_experiment = ", ".join(
            f"{e['trials_per_stimulus']:.0f}" for e in record["experiments"]
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    record["cell"],
                    "archive" if record["archived"] else "kept repo copy",
                    per_experiment,
                    str(s["n_decisions"]),
                    f"{s['n_pruned_archived']} / {s['n_decisions']}",
                    f"{s['n_prune_cluster']} ({s['n_flip_trial_vs_cluster']})",
                    f"{s['n_prune_grouped']} ({s['n_flip_trial_vs_grouped']})",
                    f"{s['n_prune_grouped_ignoring_reliability']} "
                    f"({s['n_flip_trial_vs_grouped_ignoring_reliability']})",
                    str(s["n_grouped_unreliable_decisions"]),
                    _quantile_cell(s["ratio_cluster"]),
                    _quantile_cell(s["ratio_grouped"]),
                ]
            )
            + " |"
        )
    lines.append(
        "| "
        + " | ".join(
            [
                f"**Totals** ({len(records)} cells)",
                "",
                "",
                str(totals["n_decisions"]),
                f"{totals['n_pruned_archived']} / {totals['n_decisions']}",
                f"{totals['n_prune_cluster']} ({totals['n_flip_trial_vs_cluster']})",
                f"{totals['n_prune_grouped']} ({totals['n_flip_trial_vs_grouped']})",
                f"{totals['n_prune_grouped_ignoring_reliability']} "
                f"({totals['n_flip_trial_vs_grouped_ignoring_reliability']})",
                str(totals["n_grouped_unreliable_decisions"]),
                "",
                "",
            ]
        )
        + " |"
    )
    return "\n".join(lines) + "\n"


def main(args: Args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        labels = [label.strip() for label in args.cells.split(",")] if args.cells else None
        cells = discover_cells(args.sweep, labels=labels)
        fit_kwargs = json.loads(args.fit_kwargs_json) if args.fit_kwargs_json else None
        for label, cell_dir in cells.items():
            print(f"[loo-design-effect] {label} ...", flush=True)
            record = analyze_cell(
                cell_dir,
                Path(args.work_dir),
                fit_kwargs=fit_kwargs,
                dse_multiplier=args.dse_multiplier,
            )
            record_path(out_dir, label).write_text(
                json.dumps(record, indent=1), encoding="utf-8"
            )
            s = record["summary"]
            print(
                f"[loo-design-effect] {label}: {s['n_pruned_archived']} of "
                f"{s['n_decisions']} decisions pruned; cluster flips "
                f"{s['n_flip_trial_vs_cluster']}, grouped flips "
                f"{s['n_flip_trial_vs_grouped']}; dse ratio cluster/trial median "
                f"{_quantile_cell(s['ratio_cluster'])}",
                flush=True,
            )
    table = render_table(load_records(out_dir))
    (out_dir / TABLE_NAME).write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main(tyro.cli(Args))
