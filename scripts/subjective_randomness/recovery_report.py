"""CLI: generate a markdown recovery report from a sweep root.

Reads every cell's ``holdout.csv`` final step and writes per-ground-truth
summaries (RMSE primary, KL regret secondary, Pearson r descriptive).
Optionally includes matched-seed deltas from ``compare_matched_cells``
output and oracle diagnostics from ``oracle_admitted_models`` output.

Usage:
    uv run python scripts/subjective_randomness/recovery_report.py \
        --sweep $SCRATCH/auto-psych/sweep \
        --label consolidated_raw \
        --compare vs_iter2=/path/to/vs_iter2 \
        --oracle-glob '$SCRATCH/auto-psych/sweep/run*/*/oracle.json' \
        --out $SCRATCH/auto-psych/analysis/RESULTS_auto.md
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.config import resolve_path  # noqa: E402


METRICS = ("rmse", "kl_regret", "pearson_r")
METRIC_LABELS = {"rmse": "RMSE", "kl_regret": "KL regret", "pearson_r": "Pearson r"}


@dataclass
class Args:
    """Generate a markdown recovery report from a sweep root."""

    sweep: Path
    """Sweep root directory containing run<r>/<gt>/holdout.csv."""
    label: str
    """Label for this sweep (used in the report header)."""
    out: Path
    """Output path for the markdown report. CSV written beside it."""
    compare: List[str] = field(default_factory=list)
    """Comparison dirs as '<label>=<path>' entries."""
    oracle_glob: Optional[str] = None
    """Glob pattern matching oracle.json files."""


def _read_final_step(holdout_csv: Path) -> Dict[str, Any]:
    """Read the final-step row from a holdout.csv. Legacy files without
    new columns report them as 'n/a'."""
    with holdout_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty holdout.csv at {holdout_csv}")

    final = max(rows, key=lambda r: int(r.get("global_step", 0)))

    result: Dict[str, Any] = {
        "gt_model": final["gt_model"],
        "best_model": final["best_model"],
        "global_step": int(final.get("global_step", 0)),
    }
    for metric in METRICS:
        val = final.get(metric)
        if val is None or val == "":
            result[metric] = "n/a"
        else:
            try:
                result[metric] = float(val)
            except (ValueError, TypeError):
                result[metric] = "n/a"

    return result


def _discover_cells(sweep_root: Path) -> Dict[str, Path]:
    """Map 'run<r>/<gt>' -> cell dir for every cell with a holdout.csv."""
    cells: Dict[str, Path] = {}
    for holdout_csv in sorted(sweep_root.glob("run*/*/holdout.csv")):
        cell_dir = holdout_csv.parent
        key = f"{cell_dir.parent.name}/{cell_dir.name}"
        cells[key] = cell_dir
    return cells


def _parse_compare_args(compare_list: List[str]) -> Dict[str, Path]:
    """Parse '--compare label=path' entries into a dict."""
    result: Dict[str, Path] = {}
    for entry in compare_list:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --compare format: {entry!r}. Expected 'label=path'."
            )
        label, path_str = entry.split("=", 1)
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(
                f"Compare directory does not exist: {path}"
            )
        result[label] = path
    return result


def _read_paired_csv(paired_dir: Path) -> List[Dict[str, Any]]:
    """Read paired.csv from a compare_matched_cells output dir."""
    paired_csv = paired_dir / "paired.csv"
    if not paired_csv.exists():
        raise FileNotFoundError(f"No paired.csv at {paired_csv}")

    with paired_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _read_oracle_files(oracle_glob: str) -> List[Dict[str, Any]]:
    """Read all oracle.json files matching the glob."""
    paths = sorted(glob.glob(oracle_glob))
    results = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        results.append(data)
    return results


def _fmt(val: Any, precision: int = 6) -> str:
    if val == "n/a" or val is None:
        return "n/a"
    return f"{val:.{precision}f}"


def _compute_summary(
    values: List[float], precision: int = 6
) -> Dict[str, str]:
    if not values:
        return {"mean": "n/a", "sd": "n/a", "median": "n/a",
                "min": "n/a", "max": "n/a", "n": "0"}
    arr = np.array(values)
    return {
        "mean": f"{arr.mean():.{precision}f}",
        "sd": f"{arr.std(ddof=1):.{precision}f}" if len(arr) > 1 else "n/a",
        "median": f"{np.median(arr):.{precision}f}",
        "min": f"{arr.min():.{precision}f}",
        "max": f"{arr.max():.{precision}f}",
        "n": str(len(arr)),
    }


def main(args: Args) -> None:
    sweep_root = resolve_path(args.sweep)
    if not sweep_root.exists():
        raise FileNotFoundError(f"Sweep root does not exist: {sweep_root}")

    out_path = resolve_path(args.out)

    compare_dirs = _parse_compare_args(args.compare)

    cells = _discover_cells(sweep_root)
    if not cells:
        raise ValueError(f"No cells found in {sweep_root}")

    # Read final-step data from each cell
    cell_rows: List[Dict[str, Any]] = []
    for key, cell_dir in sorted(cells.items()):
        run_label = key.split("/", 1)[0]
        gt_model = key.split("/", 1)[1]
        final = _read_final_step(cell_dir / "holdout.csv")
        cell_rows.append({
            "cell": key,
            "run": run_label,
            "gt_model": gt_model,
            **{m: final[m] for m in METRICS},
            "best_model": final["best_model"],
            "global_step": final["global_step"],
        })

    # Group by ground truth
    gt_models = sorted(set(r["gt_model"] for r in cell_rows))

    # Compute per-GT summaries
    gt_summaries: Dict[str, Dict[str, Dict[str, str]]] = {}
    for gt in gt_models:
        gt_rows = [r for r in cell_rows if r["gt_model"] == gt]
        gt_summaries[gt] = {}
        for metric in METRICS:
            values = [r[metric] for r in gt_rows if r[metric] != "n/a"]
            gt_summaries[gt][metric] = _compute_summary(values)

    # Read comparison data
    compare_data: Dict[str, Dict[str, Any]] = {}
    for label, paired_dir in compare_dirs.items():
        paired_rows = _read_paired_csv(paired_dir)
        per_gt: Dict[str, Dict[str, Any]] = {}
        for gt in gt_models:
            gt_pairs = [r for r in paired_rows if r.get("gt_model") == gt]
            per_gt[gt] = {"n_pairs": len(gt_pairs)}
            for metric in METRICS:
                delta_key = f"delta_{metric}"
                deltas = [
                    r[delta_key] for r in gt_pairs
                    if isinstance(r.get(delta_key), (int, float))
                ]
                per_gt[gt][metric] = _compute_summary(deltas)
        compare_data[label] = {"rows": paired_rows, "per_gt": per_gt}

    # Read oracle diagnostics
    oracle_data: Dict[str, Dict[str, Any]] = {}
    if args.oracle_glob:
        oracle_files = _read_oracle_files(args.oracle_glob)
        for gt in gt_models:
            selection_cells = []
            discovery_cells = []
            retention_events: List[Dict[str, Any]] = []
            for ofile in oracle_files:
                for step in ofile.get("steps", []):
                    if step.get("gt_model") != gt:
                        continue
                    gap = step.get("oracle_incumbent_gap")
                    oracle_rmse = step.get("oracle_rmse")
                    if gap is not None and gap > 0.02:
                        if oracle_rmse is not None and oracle_rmse > 0.08:
                            discovery_cells.append(step)
                        else:
                            selection_cells.append(step)
                for lost in ofile.get("lost_incumbents", []):
                    retention_events.append(lost)

            oracle_data[gt] = {
                "selection_count": len(selection_cells),
                "selection_cells": selection_cells,
                "discovery_count": len(discovery_cells),
                "discovery_cells": discovery_cells,
                "retention_count": len(retention_events),
                "retention_events": retention_events,
            }

    # --- Write markdown ---
    lines: List[str] = []
    lines.append(f"# Recovery report: {args.label}")
    lines.append("")
    lines.append(f"Sweep root: `{sweep_root}`")
    lines.append(f"Cells: {len(cells)} ({len(gt_models)} ground truths × "
                 f"{len(cells) // max(len(gt_models), 1)} repeats)")
    lines.append("")

    # Per-GT recovery table
    lines.append("## Recovery (primary: RMSE)")
    lines.append("")
    lines.append("| Ground truth | RMSE mean ± sd | median | min | max | n |"
                 " KL regret mean ± sd | Pearson r mean ± sd |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for gt in gt_models:
        s = gt_summaries[gt]
        rmse_s = s["rmse"]
        kl_s = s["kl_regret"]
        r_s = s["pearson_r"]
        lines.append(
            f"| {gt} "
            f"| {rmse_s['mean']} ± {rmse_s['sd']} "
            f"| {rmse_s['median']} | {rmse_s['min']} | {rmse_s['max']} "
            f"| {rmse_s['n']} "
            f"| {kl_s['mean']} ± {kl_s['sd']} "
            f"| {r_s['mean']} ± {r_s['sd']} |"
        )
    lines.append("")

    # Best model per cell
    lines.append("## Best model per cell")
    lines.append("")
    lines.append("| Cell | Best model | RMSE | KL regret | Pearson r |")
    lines.append("|---|---|---|---|---|")
    for row in cell_rows:
        lines.append(
            f"| {row['cell']} | {row['best_model']} "
            f"| {_fmt(row['rmse'])} | {_fmt(row['kl_regret'])} "
            f"| {_fmt(row['pearson_r'])} |"
        )
    lines.append("")

    # Matched-seed deltas
    for label, cdata in compare_data.items():
        lines.append(f"## Matched-seed deltas: {label}")
        lines.append("")
        lines.append("**Note:** These comparisons are confounded — the earlier sweeps "
                     "supplied engineered features; this one does not, plus every loop "
                     "change and the manifest scrub. Report as reference, not ablation.")
        lines.append("")
        lines.append("| Ground truth | Delta RMSE mean ± sd | median | min | max | n |"
                     " Delta KL mean ± sd | Delta r mean ± sd |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for gt in gt_models:
            gt_s = cdata["per_gt"].get(gt, {})
            for metric_key in METRICS:
                ms = gt_s.get(metric_key, {})
            rmse_s = gt_s.get("rmse", {})
            kl_s = gt_s.get("kl_regret", {})
            r_s = gt_s.get("pearson_r", {})
            lines.append(
                f"| {gt} "
                f"| {rmse_s.get('mean', 'n/a')} ± {rmse_s.get('sd', 'n/a')} "
                f"| {rmse_s.get('median', 'n/a')} "
                f"| {rmse_s.get('min', 'n/a')} "
                f"| {rmse_s.get('max', 'n/a')} "
                f"| {rmse_s.get('n', '0')} "
                f"| {kl_s.get('mean', 'n/a')} ± {kl_s.get('sd', 'n/a')} "
                f"| {r_s.get('mean', 'n/a')} ± {r_s.get('sd', 'n/a')} |"
            )

        # Per-cell rows
        lines.append("")
        lines.append("### Per-cell deltas")
        lines.append("")
        lines.append("| Cell | Delta RMSE | Delta KL | Delta r |")
        lines.append("|---|---|---|---|")
        for row in cdata["rows"]:
            lines.append(
                f"| {row.get('cell', '')} "
                f"| {_fmt(row.get('delta_rmse'))} "
                f"| {_fmt(row.get('delta_kl_regret'))} "
                f"| {_fmt(row.get('delta_pearson_r'))} |"
            )
        lines.append("")

    # Oracle diagnostics
    if oracle_data:
        lines.append("## Three-bucket diagnostic (oracle)")
        lines.append("")
        lines.append("| Ground truth | Selection failures (gap > 0.02) "
                     "| Discovery failures | Lost-incumbent (retention) |")
        lines.append("|---|---|---|---|")
        for gt in gt_models:
            od = oracle_data.get(gt, {})
            lines.append(
                f"| {gt} "
                f"| {od.get('selection_count', 0)} "
                f"| {od.get('discovery_count', 0)} "
                f"| {od.get('retention_count', 0)} |"
            )
        lines.append("")

        for gt in gt_models:
            od = oracle_data.get(gt, {})
            if od.get("selection_cells"):
                lines.append(f"### {gt} — selection failures")
                for s in od["selection_cells"]:
                    lines.append(
                        f"- exp{s['experiment']} step{s['step']}: "
                        f"oracle={s['oracle_best_model']} "
                        f"(RMSE {_fmt(s['oracle_rmse'], 4)}), "
                        f"incumbent={s['incumbent_model']} "
                        f"(RMSE {_fmt(s['incumbent_rmse'], 4)}), "
                        f"gap={_fmt(s['oracle_incumbent_gap'], 4)}"
                    )
                lines.append("")
            if od.get("discovery_cells"):
                lines.append(f"### {gt} — discovery failures")
                for s in od["discovery_cells"]:
                    lines.append(
                        f"- exp{s['experiment']} step{s['step']}: "
                        f"oracle-best RMSE {_fmt(s['oracle_rmse'], 4)} "
                        f"(no strong model discovered)"
                    )
                lines.append("")
            if od.get("retention_events"):
                lines.append(f"### {gt} — lost incumbents (retention)")
                for r in od["retention_events"]:
                    lines.append(
                        f"- {r['model']}: {r['outcome']} ({r.get('detail', '')})"
                    )
                lines.append("")

    # Write markdown
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Write CSV beside the markdown
    csv_path = out_path.with_suffix(".csv")
    csv_columns = ["cell", "run", "gt_model"] + list(METRICS) + [
        "best_model", "global_step"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(cell_rows)

    print(f"Wrote recovery report to {out_path}")
    print(f"Wrote CSV to {csv_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
