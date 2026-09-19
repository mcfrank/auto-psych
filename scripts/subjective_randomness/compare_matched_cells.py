"""CLI: compare two sweep roots on a common held-out pool.

For every ``run<r>/<gt>`` present in both sweeps: build the exhaustive pool
over lengths 1..8 minus the *union* of both cells' trained pairs, re-score
both cells via ``reevaluate_trajectories``, and write ``paired.csv`` /
``paired.json`` with the final-step RMSE, KL regret, Pearson r for each cell
and their deltas. No bootstrap over stimuli: the repeat is the stochastic unit.

Usage:
    uv run python scripts/subjective_randomness/compare_matched_cells.py \\
        --sweep-a $SCRATCH/auto-psych/sweep_featurized \\
        --sweep-b $SCRATCH/auto-psych/sweep_raw \\
        --out     $SCRATCH/auto-psych/paired_comparison
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.cell_archive import (  # noqa: E402
    CellArchiveManager,
    resolve_run_root,
)
from src.subjective_randomness.holdout_eval import (  # noqa: E402
    TRAJECTORY_COLUMNS,
    _unordered_pair,
    build_eval_stimuli,
    collect_trained_pairs,
    evaluate_trajectory,
    reevaluate_trajectories,
)
from src.subjective_randomness.holdout_recovery import (  # noqa: E402
    trajectory_tidy_rows,
)
from src.subjective_randomness.config import resolve_path  # noqa: E402


EVAL_LENGTHS = list(range(1, 9))
METRICS = ("rmse", "kl_regret", "pearson_r")


@dataclass
class Args:
    """Compare two matched-seed sweep roots on a common held-out pool."""

    sweep_a: Path
    """First sweep root (e.g. featurized)."""
    sweep_b: Path
    """Second sweep root (e.g. raw)."""
    out: Path
    """Output directory for paired.csv / paired.json."""
    cell_a: Optional[str] = None
    """Restrict to a single cell in sweep A (e.g. 'run1/falk_konold_dp')."""
    cell_b: Optional[str] = None
    """Restrict to a single cell in sweep B (matches cell_a if omitted)."""


def _discover_cells(sweep_root: Path) -> Dict[str, Path]:
    """Map 'run<r>/<gt>' -> cell dir for every cell with a holdout.json."""
    cells: Dict[str, Path] = {}
    for holdout_json in sorted(sweep_root.glob("run*/*/holdout.json")):
        cell_dir = holdout_json.parent
        key = f"{cell_dir.parent.name}/{cell_dir.name}"
        cells[key] = cell_dir
    return cells


def _load_result(cell_dir: Path) -> Dict[str, Any]:
    path = cell_dir / "holdout.json"
    if not path.exists():
        raise FileNotFoundError(f"No holdout.json at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _common_eval_pool(
    result_a: Mapping[str, Any],
    result_b: Mapping[str, Any],
    gt_model: str,
    *,
    run_root_a: Path,
    run_root_b: Path,
) -> list[dict[str, str]]:
    """Build the exhaustive eval pool minus the union of both cells' training."""
    n_exp_a = int(result_a["n_experiments"])
    n_exp_b = int(result_b["n_experiments"])

    trained_a = collect_trained_pairs(run_root_a, n_exp_a)
    trained_b = collect_trained_pairs(run_root_b, n_exp_b)

    info = build_eval_stimuli(
        run_root_a,
        n_experiments=n_exp_a,
        n_pairs=0,
        lengths=EVAL_LENGTHS,
        seed=0,
        min_remaining=1,
        exhaustive=True,
        extra_excluded_pairs=trained_b,
    )
    return info["stimuli"]


def _final_row(result: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The final trajectory step for the first gt_run in a result."""
    gt_runs = result.get("gt_runs", [])
    if not gt_runs:
        return None
    trajectory = gt_runs[0].get("trajectory", [])
    if not trajectory:
        return None
    return max(trajectory, key=lambda r: r.get("global_step", 0))


def _rescore_cell(
    result: Mapping[str, Any],
    common_pool: Sequence[Mapping[str, str]],
    other_trained: Set[Tuple[str, str]],
    seed_models_dir: Path,
    cache_dir: Optional[Path],
) -> Dict[str, Any]:
    """Re-evaluate a cell's trajectory on the common pool."""
    return reevaluate_trajectories(
        result,
        seed_models_dir=seed_models_dir,
        cache_dir=cache_dir,
        eval_pool_override={
            "lengths": EVAL_LENGTHS,
            "exhaustive": True,
            "n_pairs": 0,
            "seed": 0,
            "min_remaining": 1,
            "predict_max_draws": result.get("eval_pool", {}).get("predict_max_draws"),
        },
        extra_excluded_pairs=other_trained,
    )


def _resolve_result_run_roots(
    result: Dict[str, Any],
    cell_dir: Path,
    manager: CellArchiveManager,
) -> Dict[str, Any]:
    """Return a copy of ``result`` with each gt_run's run_root resolved.

    If the recorded ``run_root`` does not exist on disk, the cell's
    ``agent_runs.tar.gz`` is extracted and the path is rewritten.
    """
    new_runs = []
    for gt_run in result["gt_runs"]:
        gt_name = gt_run["gt_model"]
        resolved = resolve_run_root(
            cell_dir, gt_run["run_root"], gt_name, manager,
        )
        new_runs.append({**gt_run, "run_root": str(resolved)})
    return {**result, "gt_runs": new_runs}


def main(args: Args) -> None:
    sweep_a = resolve_path(args.sweep_a)
    sweep_b = resolve_path(args.sweep_b)
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells_a = _discover_cells(sweep_a)
    cells_b = _discover_cells(sweep_b)

    if args.cell_a is not None:
        cells_a = {args.cell_a: cells_a[args.cell_a]}
        cell_b_key = args.cell_b or args.cell_a
        cells_b = {cell_b_key: cells_b[cell_b_key]}

    common_keys = sorted(set(cells_a) & set(cells_b))
    if not common_keys:
        raise SystemExit(
            f"No common cells between {sweep_a} and {sweep_b}. "
            f"A has: {sorted(cells_a)}; B has: {sorted(cells_b)}"
        )

    pairs: List[Dict[str, Any]] = []
    unreconstructable: List[str] = []

    with CellArchiveManager() as archive_mgr:
        for key in common_keys:
            cell_dir_a = cells_a[key]
            cell_dir_b = cells_b[key]
            gt_model = key.split("/", 1)[1]
            run_label = key.split("/", 1)[0]

            try:
                result_a = _load_result(cell_dir_a)
                result_b = _load_result(cell_dir_b)
            except FileNotFoundError as exc:
                unreconstructable.append(f"{key}: {exc}")
                continue

            try:
                result_a = _resolve_result_run_roots(result_a, cell_dir_a, archive_mgr)
                result_b = _resolve_result_run_roots(result_b, cell_dir_b, archive_mgr)
            except (FileNotFoundError, Exception) as exc:
                unreconstructable.append(f"{key}: archive extraction failed: {exc}")
                continue

            run_root_a = Path(result_a["gt_runs"][0]["run_root"])
            run_root_b = Path(result_b["gt_runs"][0]["run_root"])

            try:
                common_pool = _common_eval_pool(
                    result_a, result_b, gt_model,
                    run_root_a=run_root_a, run_root_b=run_root_b,
                )
            except (FileNotFoundError, ValueError) as exc:
                unreconstructable.append(f"{key}: common pool failed: {exc}")
                continue

            seed_dir_a = Path(result_a["seed_models_dir"])
            seed_dir_b = Path(result_b["seed_models_dir"])
            cache_a = cell_dir_a / "mcmc_cache"
            if not cache_a.is_dir():
                cache_a = run_root_a.parent / "mcmc_cache"
            cache_b = cell_dir_b / "mcmc_cache"
            if not cache_b.is_dir():
                cache_b = run_root_b.parent / "mcmc_cache"

            trained_a = collect_trained_pairs(run_root_a, int(result_a["n_experiments"]))
            trained_b = collect_trained_pairs(run_root_b, int(result_b["n_experiments"]))

            try:
                rescored_a = _rescore_cell(result_a, common_pool, trained_b, seed_dir_a, cache_a)
                rescored_b = _rescore_cell(result_b, common_pool, trained_a, seed_dir_b, cache_b)
            except Exception as exc:
                unreconstructable.append(f"{key}: rescore failed: {exc}")
                continue

            final_a = _final_row(rescored_a)
            final_b = _final_row(rescored_b)
            if final_a is None or final_b is None:
                unreconstructable.append(f"{key}: no final trajectory row")
                continue

            pair_entry: Dict[str, Any] = {
                "cell": key,
                "gt_model": gt_model,
                "run": run_label,
                "n_eval_stimuli": len(common_pool),
            }
            for metric in METRICS:
                val_a = final_a.get(metric)
                val_b = final_b.get(metric)
                pair_entry[f"{metric}_a"] = val_a
                pair_entry[f"{metric}_b"] = val_b
                if val_a is not None and val_b is not None:
                    pair_entry[f"delta_{metric}"] = val_b - val_a
                else:
                    pair_entry[f"delta_{metric}"] = None
            pairs.append(pair_entry)

    per_gt: Dict[str, Dict[str, Any]] = {}
    gt_models = sorted(set(p["gt_model"] for p in pairs))
    for gt in gt_models:
        gt_pairs = [p for p in pairs if p["gt_model"] == gt]
        gt_summary: Dict[str, Any] = {"n_repeats": len(gt_pairs)}
        for metric in METRICS:
            deltas = [
                p[f"delta_{metric}"]
                for p in gt_pairs
                if p.get(f"delta_{metric}") is not None
            ]
            if deltas:
                arr = np.array(deltas)
                gt_summary[f"mean_delta_{metric}"] = float(arr.mean())
                gt_summary[f"median_delta_{metric}"] = float(np.median(arr))
                gt_summary[f"min_delta_{metric}"] = float(arr.min())
                gt_summary[f"max_delta_{metric}"] = float(arr.max())
            else:
                gt_summary[f"mean_delta_{metric}"] = None
                gt_summary[f"median_delta_{metric}"] = None
                gt_summary[f"min_delta_{metric}"] = None
                gt_summary[f"max_delta_{metric}"] = None
        per_gt[gt] = gt_summary

    output = {
        "sweep_a": str(sweep_a),
        "sweep_b": str(sweep_b),
        "n_pairs": len(pairs),
        "pairs": pairs,
        "per_gt": per_gt,
        "unreconstructable": unreconstructable,
    }

    (out_dir / "paired.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    csv_columns = ["cell", "gt_model", "run", "n_eval_stimuli"]
    for metric in METRICS:
        csv_columns.extend([f"{metric}_a", f"{metric}_b", f"delta_{metric}"])
    with (out_dir / "paired.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairs)

    print(f"Wrote {len(pairs)} paired comparisons to {out_dir}")
    if unreconstructable:
        print(f"  {len(unreconstructable)} cell(s) could not be reconstructed:")
        for u in unreconstructable:
            print(f"    {u}")
    for gt, summary in per_gt.items():
        print(f"  {gt}: delta RMSE mean={summary['mean_delta_rmse']}, "
              f"delta KL mean={summary['mean_delta_kl_regret']}, "
              f"delta r mean={summary['mean_delta_pearson_r']}")

    if not pairs:
        raise SystemExit(
            f"No paired cells produced. {len(unreconstructable)} cell(s) "
            f"could not be reconstructed."
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
