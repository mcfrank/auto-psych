"""CLI: generate a markdown recovery report from a sweep root.

Reads every complete cell's (one with ``holdout.json``) ``holdout.csv`` final
step — the end of its final experiment — and the fitted-seed baseline fit on
the same data, and writes per-ground-truth summaries (RMSE primary, KL regret
secondary, Pearson r descriptive) over the same cells for the loop and the
baseline. The expected cells that are partial or missing, and the complete
ones left out of the table, are listed with the reason (pass ``--n-repeats``
and ``--gt-models`` to state the expected grid).
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
from src.subjective_randomness.sweep_cells import (  # noqa: E402
    accounting_lines,
    survey_sweep,
)


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
    n_repeats: Optional[int] = None
    """The sweep's repeat count, so cells that never started are listed as
    missing (default: inferred from the run<r>/ directories, and said so)."""
    gt_models: Optional[str] = None
    """The sweep's ground truths, space-separated (as GT_MODELS; default:
    inferred from the directories present, and said so)."""


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


# The fitted-seed baseline's field for each metric it has (it has no KL regret).
FITTED_FIELDS = {"rmse": "elpd_best_rmse", "pearson_r": "elpd_best_r"}


def _fitted_baseline_at_end(cell_dir: Path) -> Dict[str, Any]:
    """The cell's fitted-seed baseline at the end of its final experiment
    (``fitted_baseline``: the ELPD-best trusted seed fit on all the data), as
    ``{"rmse", "pearson_r", "reason"}``; a value it lacks is ``"n/a"``."""
    result = json.loads((cell_dir / "holdout.json").read_text(encoding="utf-8"))
    (gt_run,) = result["gt_runs"]
    baseline = gt_run.get("fitted_baseline") or {}
    values: Dict[str, Any] = {}
    for metric, field in FITTED_FIELDS.items():
        value = baseline.get(field)
        values[metric] = float(value) if isinstance(value, (int, float)) else "n/a"
    values["reason"] = (
        baseline.get("elpd_best_reason")
        or ("" if all(values[m] != "n/a" for m in FITTED_FIELDS)
            else "not recorded (scored before the ELPD-best baseline existed)")
    )
    return values


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

    # Every expected cell: a cell without a result is listed, never silently
    # absent (it used to be: only cells with a holdout.csv were seen).
    survey = survey_sweep(
        sweep_root,
        n_repeats=args.n_repeats,
        gt_models=args.gt_models.split() if args.gt_models else None,
    )
    cells = survey.complete
    if not cells:
        raise ValueError(f"No complete cells (holdout.json) in {sweep_root}")

    # Read final-step data from each cell: the end of its final experiment,
    # with the fitted-seed baseline fit on the same (final) data.
    cell_rows: List[Dict[str, Any]] = []
    for key, cell_dir in sorted(cells.items()):
        run_label = key.split("/", 1)[0]
        gt_model = key.split("/", 1)[1]
        if not (cell_dir / "holdout.csv").exists():
            raise FileNotFoundError(f"{key} has a holdout.json but no holdout.csv")
        final = _read_final_step(cell_dir / "holdout.csv")
        fitted = _fitted_baseline_at_end(cell_dir)
        cell_rows.append({
            "cell": key,
            "run": run_label,
            "gt_model": gt_model,
            **{m: final[m] for m in METRICS},
            "best_model": final["best_model"],
            "global_step": final["global_step"],
            **{f"fitted_{m}": fitted[m] for m in FITTED_FIELDS},
            "fitted_reason": fitted["reason"],
        })

    # Group by ground truth (every expected one, even with no complete cell)
    gt_models = sorted(
        {r["gt_model"] for r in cell_rows}
        | {label.split("/", 1)[1] for label in (*survey.partial, *survey.missing)}
    )

    # The headline covers the same cells for the loop and the baseline: those
    # where every reported value is defined. The others are listed with why.
    reported = [*METRICS, *(f"fitted_{m}" for m in FITTED_FIELDS)]
    left_out: Dict[str, str] = {}
    for row in cell_rows:
        undefined = [key for key in reported if row[key] == "n/a"]
        if any(key.startswith("fitted_") for key in undefined):
            left_out[row["cell"]] = f"no fitted-seed baseline: {row['fitted_reason']}"
        elif undefined:
            left_out[row["cell"]] = f"undefined {', '.join(undefined)}"
    common = [r for r in cell_rows if r["cell"] not in left_out]

    # Compute per-GT summaries
    gt_summaries: Dict[str, Dict[str, Dict[str, str]]] = {}
    for gt in gt_models:
        gt_rows = [r for r in common if r["gt_model"] == gt]
        gt_summaries[gt] = {}
        for metric in reported:
            gt_summaries[gt][metric] = _compute_summary([r[metric] for r in gt_rows])

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
    expected_by_gt = {
        gt: sum(1 for label in (*survey.complete, *survey.partial, *survey.missing)
                if label.split("/", 1)[1] == gt)
        for gt in gt_models
    }
    lines.append(
        "Complete cells per ground truth: "
        + "; ".join(
            f"{gt}: {sum(1 for r in cell_rows if r['gt_model'] == gt)} of {expected_by_gt[gt]}"
            for gt in gt_models
        )
    )
    lines.append("")
    lines += accounting_lines(survey, included=[r["cell"] for r in common])
    if left_out:
        lines.append("Complete cells left out of the table (the loop and the baseline "
                     "are summarised over the same cells):")
        lines.append("")
        lines += [f"- `{cell}`: {reason}" for cell, reason in sorted(left_out.items())]
        lines.append("")

    # Per-GT recovery table
    lines.append("## Recovery at the end of the final experiment (primary: RMSE)")
    lines.append("")
    lines.append("The fitted-seed baseline is the ELPD-best trusted seed fit on the "
                 "same final data; both columns cover the same cells (n).")
    lines.append("")
    lines.append("| Ground truth | RMSE mean ± sd | median | min | max | n |"
                 " fitted-seed RMSE mean ± sd | KL regret mean ± sd |"
                 " Pearson r mean ± sd | fitted-seed r mean ± sd |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for gt in gt_models:
        s = gt_summaries[gt]
        rmse_s = s["rmse"]
        kl_s = s["kl_regret"]
        r_s = s["pearson_r"]
        fitted_rmse = s["fitted_rmse"]
        fitted_r = s["fitted_pearson_r"]
        lines.append(
            f"| {gt} "
            f"| {rmse_s['mean']} ± {rmse_s['sd']} "
            f"| {rmse_s['median']} | {rmse_s['min']} | {rmse_s['max']} "
            f"| {rmse_s['n']} "
            f"| {fitted_rmse['mean']} ± {fitted_rmse['sd']} "
            f"| {kl_s['mean']} ± {kl_s['sd']} "
            f"| {r_s['mean']} ± {r_s['sd']} "
            f"| {fitted_r['mean']} ± {fitted_r['sd']} |"
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
        "best_model", "global_step",
        *(f"fitted_{m}" for m in FITTED_FIELDS), "fitted_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(cell_rows)

    print(f"Wrote recovery report to {out_path}")
    print(f"Wrote CSV to {csv_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
