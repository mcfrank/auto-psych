"""Tests for scripts/subjective_randomness/recovery_report.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest


HOLDOUT_CSV_HEADER = (
    "gt_model,experiment,step,iteration,global_step,best_model,"
    "pearson_r,rmse,kl_regret,bias,calib_slope,calib_intercept,"
    "pearson_r_bma,rmse_bma,kl_regret_bma,bias_bma,"
    "calib_slope_bma,calib_intercept_bma"
)

LEGACY_CSV_HEADER = (
    "gt_model,experiment,step,iteration,global_step,best_model,"
    "pearson_r,rmse"
)


def _write_holdout_csv(
    cell_dir: Path,
    gt_model: str,
    best_model: str,
    rmse: float,
    kl_regret: float,
    pearson_r: float,
    global_step: int = 5,
    legacy: bool = False,
) -> None:
    """Write a minimal holdout.csv with one final-step row."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cell_dir / "holdout.csv"
    if legacy:
        lines = [
            LEGACY_CSV_HEADER,
            f"{gt_model},1,0,,0,seed_model,0.5,0.2",
            f"{gt_model},2,0,,{global_step},{best_model},{pearson_r},{rmse}",
        ]
    else:
        lines = [
            HOLDOUT_CSV_HEADER,
            (
                f"{gt_model},1,0,,0,seed_model,0.5,0.2,0.1,0.01,0.9,0.05,"
                "0.5,0.2,0.1,0.01,0.9,0.05"
            ),
            (
                f"{gt_model},2,0,,{global_step},{best_model},{pearson_r},{rmse},"
                f"{kl_regret},0.01,0.95,0.03,"
                f"{pearson_r},{rmse},{kl_regret},0.01,0.95,0.03"
            ),
        ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paired_csv(out_dir: Path, pairs: list[dict]) -> None:
    """Write a paired.csv as compare_matched_cells would."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cell", "gt_model", "run", "n_eval_stimuli",
        "rmse_a", "rmse_b", "delta_rmse",
        "kl_regret_a", "kl_regret_b", "delta_kl_regret",
        "pearson_r_a", "pearson_r_b", "delta_pearson_r",
    ]
    with (out_dir / "paired.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)


def _write_oracle_json(cell_dir: Path, gt_model: str, oracle_rmse: float,
                       incumbent_rmse: float, lost: list[dict] | None = None) -> None:
    """Write an oracle.json as oracle_admitted_models would."""
    gap = incumbent_rmse - oracle_rmse
    data = {
        "result_path": str(cell_dir / "holdout.json"),
        "steps": [{
            "gt_model": gt_model,
            "experiment": 2,
            "step": 0,
            "n_models_scored": 5,
            "oracle_best_model": "oracle_model",
            "oracle_rmse": oracle_rmse,
            "incumbent_model": "incumbent_model",
            "incumbent_rmse": incumbent_rmse,
            "oracle_incumbent_gap": gap,
            "final_model": "incumbent_model",
            "final_rmse": incumbent_rmse,
        }],
        "lost_incumbents": lost or [],
    }
    (cell_dir / "oracle.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def _build_synthetic_sweep(tmp_path: Path) -> Path:
    """2 GTs x 3 repeats with known values."""
    sweep = tmp_path / "sweep"
    gt_a = "model_alpha"
    gt_b = "model_beta"

    # run1/model_alpha: rmse=0.10, kl=0.02, r=0.90
    _write_holdout_csv(
        sweep / "run1" / gt_a, gt_a, "best_a1",
        rmse=0.10, kl_regret=0.02, pearson_r=0.90,
    )
    # run2/model_alpha: rmse=0.20, kl=0.04, r=0.80
    _write_holdout_csv(
        sweep / "run2" / gt_a, gt_a, "best_a2",
        rmse=0.20, kl_regret=0.04, pearson_r=0.80,
    )
    # run3/model_alpha: rmse=0.15, kl=0.03, r=0.85
    _write_holdout_csv(
        sweep / "run3" / gt_a, gt_a, "best_a3",
        rmse=0.15, kl_regret=0.03, pearson_r=0.85,
    )

    # run1/model_beta: rmse=0.05, kl=0.01, r=0.95
    _write_holdout_csv(
        sweep / "run1" / gt_b, gt_b, "best_b1",
        rmse=0.05, kl_regret=0.01, pearson_r=0.95,
    )
    # run2/model_beta: rmse=0.07, kl=0.015, r=0.93
    _write_holdout_csv(
        sweep / "run2" / gt_b, gt_b, "best_b2",
        rmse=0.07, kl_regret=0.015, pearson_r=0.93,
    )
    # run3/model_beta: rmse=0.03, kl=0.005, r=0.97 (legacy — no kl_regret)
    _write_holdout_csv(
        sweep / "run3" / gt_b, gt_b, "best_b3",
        rmse=0.03, kl_regret=0.005, pearson_r=0.97,
        legacy=True,
    )

    return sweep


def test_basic_report(tmp_path: Path) -> None:
    """Report on a synthetic sweep produces correct per-GT summaries."""
    from scripts.subjective_randomness.recovery_report import main, Args

    sweep = _build_synthetic_sweep(tmp_path)
    out_md = tmp_path / "report.md"
    args = Args(sweep=sweep, label="test_sweep", out=out_md)
    main(args)

    assert out_md.exists()
    md_text = out_md.read_text(encoding="utf-8")

    csv_path = out_md.with_suffix(".csv")
    assert csv_path.exists()

    with csv_path.open(encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    alpha_rows = [r for r in reader if r["gt_model"] == "model_alpha"]
    beta_rows = [r for r in reader if r["gt_model"] == "model_beta"]
    assert len(alpha_rows) == 3
    assert len(beta_rows) == 3

    # model_alpha: rmse = [0.10, 0.20, 0.15] -> mean 0.15, sd ~0.05
    alpha_rmse = [float(r["rmse"]) for r in alpha_rows]
    assert np.isclose(np.mean(alpha_rmse), 0.15)

    # model_beta: rmse = [0.05, 0.07, 0.03] -> mean 0.05
    beta_rmse = [float(r["rmse"]) for r in beta_rows]
    assert np.isclose(np.mean(beta_rmse), 0.05)

    # Legacy cell (run3/model_beta) should show n/a for kl_regret
    legacy_row = [r for r in beta_rows if r["run"] == "run3"][0]
    assert legacy_row["kl_regret"] == "n/a"

    # Non-legacy cells have kl_regret values
    normal_row = [r for r in beta_rows if r["run"] == "run1"][0]
    assert float(normal_row["kl_regret"]) == 0.01

    # Markdown contains the label
    assert "test_sweep" in md_text
    # Markdown contains gt model names
    assert "model_alpha" in md_text
    assert "model_beta" in md_text


def test_report_with_compare(tmp_path: Path) -> None:
    """Report with --compare includes per-GT delta summaries."""
    from scripts.subjective_randomness.recovery_report import main, Args

    sweep = _build_synthetic_sweep(tmp_path)

    # Build a paired comparison dir
    compare_dir = tmp_path / "vs_old"
    _write_paired_csv(compare_dir, [
        {"cell": "run1/model_alpha", "gt_model": "model_alpha", "run": "run1",
         "n_eval_stimuli": 100,
         "rmse_a": 0.15, "rmse_b": 0.10, "delta_rmse": -0.05,
         "kl_regret_a": 0.03, "kl_regret_b": 0.02, "delta_kl_regret": -0.01,
         "pearson_r_a": 0.85, "pearson_r_b": 0.90, "delta_pearson_r": 0.05},
        {"cell": "run2/model_alpha", "gt_model": "model_alpha", "run": "run2",
         "n_eval_stimuli": 100,
         "rmse_a": 0.25, "rmse_b": 0.20, "delta_rmse": -0.05,
         "kl_regret_a": 0.05, "kl_regret_b": 0.04, "delta_kl_regret": -0.01,
         "pearson_r_a": 0.75, "pearson_r_b": 0.80, "delta_pearson_r": 0.05},
    ])

    out_md = tmp_path / "report_compare.md"
    args = Args(
        sweep=sweep, label="test_sweep", out=out_md,
        compare=["vs_old=" + str(compare_dir)],
    )
    main(args)

    md_text = out_md.read_text(encoding="utf-8")
    assert "vs_old" in md_text
    # The delta should appear
    assert "delta" in md_text.lower() or "Delta" in md_text


def test_report_with_oracle(tmp_path: Path) -> None:
    """Report with --oracle-glob includes diagnostic buckets."""
    from scripts.subjective_randomness.recovery_report import main, Args

    sweep = _build_synthetic_sweep(tmp_path)

    # Add oracle files — one with a big gap (selection failure), one small
    _write_oracle_json(
        sweep / "run1" / "model_alpha", "model_alpha",
        oracle_rmse=0.02, incumbent_rmse=0.10,  # gap 0.08 > 0.02
    )
    _write_oracle_json(
        sweep / "run2" / "model_alpha", "model_alpha",
        oracle_rmse=0.18, incumbent_rmse=0.20,  # gap 0.02 <= 0.02
    )
    _write_oracle_json(
        sweep / "run1" / "model_beta", "model_beta",
        oracle_rmse=0.04, incumbent_rmse=0.05,  # gap 0.01 <= 0.02
        lost=[{"model": "lost_one", "outcome": "pruned",
               "detail": "elpd", "context": "exp2"}],
    )

    out_md = tmp_path / "report_oracle.md"
    oracle_glob = str(sweep / "run*" / "*" / "oracle.json")
    args = Args(
        sweep=sweep, label="test_sweep", out=out_md,
        oracle_glob=oracle_glob,
    )
    main(args)

    md_text = out_md.read_text(encoding="utf-8")
    # model_alpha has 1 cell with gap > 0.02
    assert "selection" in md_text.lower() or "oracle" in md_text.lower()
    # model_beta has a lost incumbent
    assert "lost" in md_text.lower() or "retention" in md_text.lower()


def test_missing_compare_dir_raises(tmp_path: Path) -> None:
    """A missing compare dir is an error, not silently skipped."""
    from scripts.subjective_randomness.recovery_report import main, Args

    sweep = _build_synthetic_sweep(tmp_path)
    out_md = tmp_path / "report_missing.md"
    args = Args(
        sweep=sweep, label="test_sweep", out=out_md,
        compare=["vs_ghost=/nonexistent/path"],
    )
    with pytest.raises((FileNotFoundError, SystemExit, ValueError)):
        main(args)


def test_missing_sweep_root_raises(tmp_path: Path) -> None:
    """A missing sweep root is an error."""
    from scripts.subjective_randomness.recovery_report import main, Args

    out_md = tmp_path / "report_nosweep.md"
    args = Args(
        sweep=Path("/nonexistent/sweep"), label="test_sweep", out=out_md,
    )
    with pytest.raises((FileNotFoundError, SystemExit, ValueError)):
        main(args)
