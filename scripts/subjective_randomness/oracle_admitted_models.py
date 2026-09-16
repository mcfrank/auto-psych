"""CLI: oracle diagnostic over all models the inner loop ever held.

For each step (default: final step of each experiment), scores every model
under ``experiment<k>/model_loop/models/*.py`` and ``models/pruned/*.py`` on
the held-out eval pool, then writes ``oracle.json``/``oracle.csv`` with:
oracle-best model and RMSE, the incumbent's RMSE, and the gap.

Trees archived as ``agent_runs.tar.gz`` are extracted into a temp dir.

Usage:
    uv run python scripts/subjective_randomness/oracle_admitted_models.py \\
        --result /path/to/holdout.json
"""

from __future__ import annotations

import csv
import json
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

import numpy as np
import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.models.pymc_inference import fit_model, make_stim_data, pm_data_inputs  # noqa: E402
from src.pipelines.inner_loop.hypothesis_ledger import (  # noqa: E402
    HypothesisLedger,
    LEDGER_FILENAME,
)
from src.subjective_randomness.holdout_recovery import (  # noqa: E402
    _unordered_pair,
    _resolve_model_dir,
    collect_trained_pairs,
    evaluate_trajectory,
    feature_rows,
    p_left_fixed_params,
)
from src.subjective_randomness.recover import pearson_r  # noqa: E402
from src.subjective_randomness.recovery_metrics import (  # noqa: E402
    kl_regret,
    rmse as compute_rmse,
)
from src.subjective_randomness.config import resolve_path  # noqa: E402


@dataclass
class Args:
    """Oracle diagnostic: score all admitted models on the held-out pool."""

    result: Path
    """Path to holdout.json."""
    steps: Literal["final", "all"] = "final"
    """Which steps to score: 'final' (last step per experiment) or 'all'."""


def _discover_models(loop_dir: Path) -> list[str]:
    """All model names under models/ and models/pruned/."""
    models_dir = loop_dir / "models"
    names: list[str] = []
    if models_dir.is_dir():
        for py_file in sorted(models_dir.glob("*.py")):
            if py_file.name != "__init__.py":
                names.append(py_file.stem)
    pruned_dir = models_dir / "pruned"
    if pruned_dir.is_dir():
        for py_file in sorted(pruned_dir.glob("*.py")):
            if py_file.name != "__init__.py":
                names.append(py_file.stem)
    return names


def _eval_prediction(
    fitted: Any,
    eval_rows: list[dict],
    participant_ids: list[Any],
    max_draws: Optional[int] = None,
) -> np.ndarray:
    stim_data = make_stim_data(fitted.model, eval_rows)
    data_inputs = pm_data_inputs(fitted.model) if fitted.model is not None else []
    has_participant = any(d.name == "participant_id" for d in data_inputs)

    if has_participant:
        per_pid: list[np.ndarray] = []
        for pid in participant_ids:
            pid_data = {**stim_data, "participant_id": [pid] * stim_data["n"]}
            pred = fitted.predict_p_left(pid_data)
            if max_draws is not None and hasattr(pred, "__len__") and len(pred) > max_draws:
                pred = pred[:max_draws]
            per_pid.append(np.asarray(pred, dtype=float))
        return np.mean(per_pid, axis=0)

    pred = fitted.predict_p_left(stim_data)
    return np.asarray(pred, dtype=float)


def _participant_ids_in(responses_csv: Path) -> list[str]:
    ids: list[str] = []
    with responses_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("participant_id")
            if pid is not None and pid not in ids:
                ids.append(pid)
    return ids


def _score_model(
    name: str,
    loop_dir: Path,
    eval_rows: list[dict],
    gt_p: np.ndarray,
    cache_dir: Optional[Path],
    fit_kwargs: dict,
    participant_ids: list[Any],
    predict_max_draws: Optional[int] = None,
) -> Optional[dict]:
    """Score a single model on the eval pool. Returns None if it fails."""
    models_dir = _resolve_model_dir(loop_dir / "models", name)
    try:
        fitted = fit_model(
            name, models_dir, loop_dir / "responses.csv",
            cache_dir=cache_dir, **fit_kwargs,
        )
        pred = _eval_prediction(
            fitted, eval_rows, participant_ids, predict_max_draws,
        )
    except Exception as exc:
        return None

    pred_list = pred.tolist()
    gt_list = gt_p.tolist()
    rmse_val = compute_rmse(gt_list, pred_list)
    r_val = pearson_r(gt_list, pred_list)
    kl_val = kl_regret(gt_list, pred_list)
    return {
        "model": name,
        "rmse": rmse_val,
        "pearson_r": r_val,
        "kl_regret": kl_val,
    }


def _find_lost_incumbents(
    history_steps: list[dict],
    ledger_path: Path,
) -> list[dict]:
    """Models that were best_model at an earlier step and were later pruned/dropped."""
    prior_bests: set[str] = set()
    lost: list[dict] = []

    for step in history_steps:
        best = step.get("best_model")
        if best:
            prior_bests.add(best)

    if not ledger_path.exists():
        return lost

    ledger = HypothesisLedger(ledger_path)
    for entry in ledger.entries():
        if entry.name in prior_bests and entry.outcome in ("pruned", "dropped"):
            lost.append({
                "model": entry.name,
                "outcome": entry.outcome,
                "detail": entry.detail,
                "context": entry.context,
            })

    return lost


def _extract_archive(cell_dir: Path) -> Optional[tempfile.TemporaryDirectory]:
    """If the run tree is archived, extract to a temp dir and return it."""
    tar_path = cell_dir / "agent_runs.tar.gz"
    if not tar_path.exists():
        return None
    td = tempfile.TemporaryDirectory(dir=cell_dir, prefix="oracle_extract_")
    with tarfile.open(tar_path) as tf:
        tf.extractall(td.name)
    return td


def main(args: Args) -> None:
    result_path = resolve_path(args.result)
    result = json.loads(result_path.read_text(encoding="utf-8"))

    n_experiments = int(result["n_experiments"])
    fit_kwargs = dict(result.get("fit_kwargs", {}))
    eval_pool = result.get("eval_pool", {})
    predict_max_draws = eval_pool.get("predict_max_draws")

    all_steps: list[dict] = []
    all_lost: list[dict] = []

    for gt_run in result["gt_runs"]:
        gt_model = gt_run["gt_model"]
        gt_params = gt_run["params"]
        run_root = Path(gt_run["run_root"])

        eval_stimuli_path = run_root / "eval_stimuli.json"
        if not eval_stimuli_path.exists():
            raise FileNotFoundError(f"No eval_stimuli.json at {eval_stimuli_path}")
        eval_stimuli = json.loads(eval_stimuli_path.read_text(encoding="utf-8"))
        eval_rows = feature_rows(eval_stimuli)

        seed_models_dir = Path(result["seed_models_dir"])
        gt_p = p_left_fixed_params(gt_model, seed_models_dir, eval_stimuli, gt_params)

        cache_dir = run_root.parent / "mcmc_cache"
        if not cache_dir.is_dir():
            cache_dir = None

        all_history: list[dict] = []
        for exp_num in range(1, n_experiments + 1):
            loop_dir = run_root / f"experiment{exp_num}" / "model_loop"
            history_path = loop_dir / "history.json"
            if not history_path.exists():
                continue
            history = json.loads(history_path.read_text(encoding="utf-8"))
            all_history.extend(history)

            if args.steps == "final":
                steps_to_score = [history[-1]] if history else []
            else:
                steps_to_score = history

            for step_entry in steps_to_score:
                model_names = _discover_models(loop_dir)
                if not model_names:
                    continue

                participant_ids = _participant_ids_in(loop_dir / "responses.csv")
                scores = []
                for name in model_names:
                    score = _score_model(
                        name, loop_dir, eval_rows, gt_p,
                        cache_dir, fit_kwargs, participant_ids, predict_max_draws,
                    )
                    if score is not None:
                        scores.append(score)

                if not scores:
                    continue

                oracle_best = min(scores, key=lambda s: s["rmse"])
                incumbent = step_entry["best_model"]
                incumbent_score = next(
                    (s for s in scores if s["model"] == incumbent), None
                )

                step_result = {
                    "gt_model": gt_model,
                    "experiment": exp_num,
                    "step": step_entry["step"],
                    "n_models_scored": len(scores),
                    "oracle_best_model": oracle_best["model"],
                    "oracle_rmse": oracle_best["rmse"],
                    "incumbent_model": incumbent,
                    "incumbent_rmse": incumbent_score["rmse"] if incumbent_score else None,
                    "oracle_incumbent_gap": (
                        (incumbent_score["rmse"] - oracle_best["rmse"])
                        if incumbent_score else None
                    ),
                    "final_model": step_entry["best_model"],
                    "final_rmse": incumbent_score["rmse"] if incumbent_score else None,
                }
                all_steps.append(step_result)

        ledger_path = (
            run_root / f"experiment{n_experiments}" / "model_loop" / LEDGER_FILENAME
        )
        lost = _find_lost_incumbents(all_history, ledger_path)
        all_lost.extend(lost)

    output = {
        "result_path": str(result_path),
        "steps": all_steps,
        "lost_incumbents": all_lost,
    }

    out_dir = result_path.parent
    oracle_json = out_dir / "oracle.json"
    oracle_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    csv_columns = [
        "gt_model", "experiment", "step", "n_models_scored",
        "oracle_best_model", "oracle_rmse",
        "incumbent_model", "incumbent_rmse",
        "oracle_incumbent_gap", "final_model", "final_rmse",
    ]
    oracle_csv = out_dir / "oracle.csv"
    with oracle_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_steps)

    print(f"Wrote oracle diagnostic to {oracle_json} and {oracle_csv}")
    print(f"  {len(all_steps)} step(s) scored")
    if all_lost:
        print(f"  {len(all_lost)} lost incumbent(s):")
        for lost in all_lost:
            print(f"    {lost['model']}: {lost['outcome']} ({lost['detail']})")


if __name__ == "__main__":
    main(tyro.cli(Args))
