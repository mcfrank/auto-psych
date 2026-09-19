"""The recovery ceiling: how well the ground truth's own model does.

Every recovery number in this campaign is an RMSE between a discovered model's
held-out ``p_left`` and the ground truth's. What none of them say is how small
that RMSE *could* be. The generating process used fixed parameters; the loop
fits parameters from a few thousand trials on an adaptively chosen design. So
even a model with the ground truth's exact functional form lands short of zero,
by an amount that is finite-data estimation error and nothing else.

That amount is the ceiling. Refit the held-out model itself on the cell's own
training responses, predict the held-out pool, and score it the same way a
discovered model is scored. The gap between the ceiling and what the loop
achieved is the part attributable to search; the ceiling itself is not a
recovery failure and must not be read as one.

Two ceilings are reported per cell:

* ``ceiling_rmse`` — the ground-truth model refit on this cell's data. The
  achievable floor for *this* cell's data and design.
* ``oracle_rmse`` — the ground-truth model at its true fixed parameters, which
  is 0 by construction. Reported only as a check that the target and the
  prediction path agree; a non-zero value means the evaluation is broken, not
  that the model is.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.models.pymc_inference import fit_model
from src.subjective_randomness.cell_archive import CellArchiveManager, resolve_run_root
from src.subjective_randomness.holdout_data import _raw_eval_rows, p_left_fixed_params
from src.subjective_randomness.holdout_eval import _eval_prediction, _participant_ids_in
from src.subjective_randomness.recovery_metrics import kl_regret, rmse

CELL_GLOB = "run*/*/holdout.json"


@dataclass
class CellCeiling:
    """One cell's ceiling, and the loop's achieved result beside it."""

    cell: str
    gt_model: str
    n_eval_stimuli: int
    ceiling_rmse: float
    ceiling_kl: float
    oracle_rmse: float
    loop_rmse: Optional[float] = None
    loop_best_model: Optional[str] = None
    gap_to_ceiling: Optional[float] = None
    """``loop_rmse - ceiling_rmse``: the part of the loop's error that is search,
    not finite data. None when the cell has no recorded final step."""

    def as_row(self) -> Dict[str, Any]:
        return {
            "cell": self.cell,
            "gt_model": self.gt_model,
            "n_eval_stimuli": self.n_eval_stimuli,
            "ceiling_rmse": self.ceiling_rmse,
            "ceiling_kl": self.ceiling_kl,
            "oracle_rmse": self.oracle_rmse,
            "loop_rmse": self.loop_rmse,
            "loop_best_model": self.loop_best_model,
            "gap_to_ceiling": self.gap_to_ceiling,
        }


@dataclass
class CeilingRun:
    """Every cell's ceiling, plus the cells that could not be scored."""

    cells: List[CellCeiling] = field(default_factory=list)
    unscored: List[Dict[str, str]] = field(default_factory=list)
    """``{"cell": ..., "reason": ...}`` — listed, never silently dropped."""


def discover_cells(sweep_root: Path) -> List[Path]:
    """Every ``run<r>/<gt>/holdout.json`` under ``sweep_root``, sorted."""
    sweep_root = Path(sweep_root)
    if not sweep_root.is_dir():
        raise FileNotFoundError(f"No sweep root at {sweep_root}")
    found = sorted(sweep_root.glob(CELL_GLOB))
    if not found:
        raise FileNotFoundError(
            f"No cells matching {CELL_GLOB!r} under {sweep_root}; is this a sweep root?"
        )
    return found


def final_step_from_csv(holdout_csv: Path) -> Optional[Dict[str, Any]]:
    """The loop's last trajectory row, or None when the CSV is absent/empty."""
    if not holdout_csv.is_file():
        return None
    with holdout_csv.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-1] if rows else None


def training_responses(run_root: Path, n_experiments: int) -> Path:
    """The CSV the loop's final models were fit on.

    The inner loop pools every experiment's responses, so the last experiment's
    ``model_loop/responses.csv`` is the full training set. Falls back to the
    evaluator's ``pooled_responses.csv``. Raises if neither exists — a ceiling
    computed on different data than the loop saw would not be a ceiling.
    """
    for exp_num in range(n_experiments, 0, -1):
        candidate = run_root / f"experiment{exp_num}" / "model_loop" / "responses.csv"
        if candidate.is_file():
            return candidate
    pooled = run_root / "pooled_responses.csv"
    if pooled.is_file():
        return pooled
    raise FileNotFoundError(
        f"No training responses under {run_root} "
        f"(looked for experiment*/model_loop/responses.csv and pooled_responses.csv)"
    )


def ceiling_for_cell(
    cell_dir: Path,
    *,
    manager: CellArchiveManager,
    fit_kwargs: Optional[Mapping[str, Any]] = None,
    predict_max_draws: Optional[int] = None,
    limit_eval: Optional[int] = None,
    gt_models_dir: Optional[Path] = None,
) -> CellCeiling:
    """Refit this cell's held-out ground truth on its own data and score it.

    ``fit_kwargs`` overrides the sampler settings recorded in the cell (tests
    pass tiny draws). ``limit_eval`` scores only the first N eval stimuli, for
    tests — a full run must leave it unset. ``gt_models_dir`` overrides the
    recorded registry path, which points into the sweep's staged harness repo
    and may have been cleaned up.
    """
    cell_dir = Path(cell_dir)
    result = json.loads((cell_dir / "holdout.json").read_text(encoding="utf-8"))
    gt_run = result["gt_runs"][0]
    gt_model = gt_run["gt_model"]
    gt_params = gt_run["params"]

    models_dir = Path(gt_models_dir or result["seed_models_dir"])
    if not (models_dir / f"{gt_model}.py").is_file():
        raise FileNotFoundError(
            f"No {gt_model}.py under {models_dir} — pass --gt-models-dir at a "
            f"registry that still holds the held-out model"
        )

    run_root = resolve_run_root(cell_dir, gt_run["run_root"], gt_model, manager)
    eval_path = run_root / "eval_stimuli.json"
    if not eval_path.is_file():
        raise FileNotFoundError(f"No eval_stimuli.json at {eval_path}")
    eval_stimuli: Sequence[Mapping[str, str]] = json.loads(
        eval_path.read_text(encoding="utf-8")
    )
    if limit_eval is not None:
        eval_stimuli = list(eval_stimuli)[:limit_eval]

    responses_path = training_responses(run_root, int(result["n_experiments"]))
    sampler = dict(result.get("fit_kwargs") or {})
    sampler.update(dict(fit_kwargs or {}))

    target = p_left_fixed_params(gt_model, models_dir, eval_stimuli, gt_params)
    eval_rows = _raw_eval_rows(eval_stimuli)

    fitted = fit_model(
        gt_model,
        models_dir,
        responses_path,
        cache_dir=cell_dir / "mcmc_cache",
        **sampler,
    )
    predicted = _eval_prediction(
        fitted,
        eval_rows,
        participant_ids=_participant_ids_in(responses_path),
        max_draws=predict_max_draws,
    )

    final = final_step_from_csv(cell_dir / "holdout.csv")
    loop_rmse = float(final["rmse"]) if final and final.get("rmse") else None
    ceiling = rmse(target.tolist(), predicted.tolist())
    return CellCeiling(
        cell=f"{cell_dir.parent.name}/{cell_dir.name}",
        gt_model=gt_model,
        n_eval_stimuli=len(eval_stimuli),
        ceiling_rmse=ceiling,
        ceiling_kl=kl_regret(target.tolist(), predicted.tolist()),
        # The target is this same model at its true parameters, so this is 0 up
        # to floating point. A non-zero value means the evaluation path is wrong.
        oracle_rmse=rmse(target.tolist(), target.tolist()),
        loop_rmse=loop_rmse,
        loop_best_model=(final or {}).get("best_model"),
        gap_to_ceiling=None if loop_rmse is None else loop_rmse - ceiling,
    )


def run_ceiling(
    sweep_root: Path,
    *,
    fit_kwargs: Optional[Mapping[str, Any]] = None,
    predict_max_draws: Optional[int] = None,
    limit_eval: Optional[int] = None,
    gt_models_dir: Optional[Path] = None,
    on_progress=print,
) -> CeilingRun:
    """Score every cell under ``sweep_root``; list the ones that cannot be."""
    out = CeilingRun()
    with CellArchiveManager() as manager:
        for holdout_json in discover_cells(sweep_root):
            cell_dir = holdout_json.parent
            label = f"{cell_dir.parent.name}/{cell_dir.name}"
            try:
                cell = ceiling_for_cell(
                    cell_dir,
                    manager=manager,
                    fit_kwargs=fit_kwargs,
                    predict_max_draws=predict_max_draws,
                    limit_eval=limit_eval,
                    gt_models_dir=gt_models_dir,
                )
            except Exception as exc:  # recorded, never silently skipped
                out.unscored.append({"cell": label, "reason": f"{type(exc).__name__}: {exc}"})
                if on_progress:
                    on_progress(f"  [unscored] {label}: {type(exc).__name__}: {exc}")
                continue
            out.cells.append(cell)
            if on_progress:
                on_progress(
                    f"  {label}: ceiling {cell.ceiling_rmse:.4f}  "
                    f"loop {cell.loop_rmse if cell.loop_rmse is None else round(cell.loop_rmse, 4)}  "
                    f"gap {cell.gap_to_ceiling if cell.gap_to_ceiling is None else round(cell.gap_to_ceiling, 4)}"
                )
    return out


def summarize_by_ground_truth(run: CeilingRun) -> List[Dict[str, Any]]:
    """Per ground truth: mean ceiling, mean loop RMSE, mean gap, n."""
    by_gt: Dict[str, List[CellCeiling]] = {}
    for cell in run.cells:
        by_gt.setdefault(cell.gt_model, []).append(cell)
    rows: List[Dict[str, Any]] = []
    for gt_model, cells in sorted(by_gt.items()):
        ceilings = [c.ceiling_rmse for c in cells]
        gaps = [c.gap_to_ceiling for c in cells if c.gap_to_ceiling is not None]
        loops = [c.loop_rmse for c in cells if c.loop_rmse is not None]
        rows.append(
            {
                "gt_model": gt_model,
                "n": len(cells),
                "mean_ceiling_rmse": float(np.mean(ceilings)),
                "max_ceiling_rmse": float(np.max(ceilings)),
                "mean_loop_rmse": float(np.mean(loops)) if loops else None,
                "mean_gap": float(np.mean(gaps)) if gaps else None,
            }
        )
    return rows
