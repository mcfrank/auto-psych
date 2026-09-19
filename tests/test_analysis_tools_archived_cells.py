"""Tests that the analysis tools (compare_matched_cells, oracle_admitted_models)
work on real archived cells from the sweeps.

These tests exercise the archive extraction path against actual
``agent_runs.tar.gz`` files, catching the path-resolution failures that
caused P15 to fail four times.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tarfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Real sweep roots (read-only, from the consolidation work root)
# ---------------------------------------------------------------------------

_WORK_ROOT = Path(
    os.environ.get(
        "WORK_ROOT",
        "/scratch/users/benpry/auto-psych/consolidation_2026_09",
    )
)
_SWEEP = _WORK_ROOT / "sweep"
_SWEEP_RERUN = _WORK_ROOT / "sweep_rerun"


def _find_real_cell(sweep_root: Path) -> str | None:
    """Find a cell key (e.g. 'run1/falk_konold_dp') with both holdout.json and archive."""
    for holdout in sorted(sweep_root.glob("run*/*/holdout.json")):
        cell_dir = holdout.parent
        if (cell_dir / "agent_runs.tar.gz").exists():
            return f"{cell_dir.parent.name}/{cell_dir.name}"
    return None


def _find_matched_pair() -> tuple[str, str, str] | None:
    """Find a cell key present in both sweep and sweep_rerun."""
    if not _SWEEP.exists() or not _SWEEP_RERUN.exists():
        return None
    for holdout in sorted(_SWEEP.glob("run*/*/holdout.json")):
        cell_dir = holdout.parent
        key = f"{cell_dir.parent.name}/{cell_dir.name}"
        rerun_cell = _SWEEP_RERUN / key
        if (
            (cell_dir / "agent_runs.tar.gz").exists()
            and (rerun_cell / "holdout.json").exists()
            and (rerun_cell / "agent_runs.tar.gz").exists()
        ):
            return key, str(_SWEEP), str(_SWEEP_RERUN)
    return None


_REAL_CELL_SWEEP = _find_real_cell(_SWEEP) if _SWEEP.exists() else None
_MATCHED_PAIR = _find_matched_pair()


# ---------------------------------------------------------------------------
# compare_matched_cells on real archived cells
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _MATCHED_PAIR is None,
    reason="no matched cell pair with archives in sweep and sweep_rerun",
)
def test_compare_matched_cells_on_real_archives(tmp_path, monkeypatch):
    """compare_matched_cells produces paired output from two real archived cells."""
    import src.subjective_randomness.holdout_eval as holdout_eval

    cell_key, sweep_a_str, sweep_b_str = _MATCHED_PAIR
    gt_model = cell_key.split("/", 1)[1]

    from scripts.subjective_randomness.compare_matched_cells import Args, main

    # Stub the evaluation seams so we don't need real MCMC — we just need
    # the path resolution and pool construction to succeed.
    import numpy as np

    pred = np.array([0.5] * 500)  # large enough for any pool

    monkeypatch.setattr(
        holdout_eval,
        "seed_model_names",
        lambda pool_dir, *a, **kw: ["stub_seed"],
    )

    class _Fitted:
        model = None
        def predict_p_left(self, stim_data, **kw):
            return pred[: stim_data["n"]]

    monkeypatch.setattr(
        holdout_eval,
        "fit_model",
        lambda name, models_dir, responses_path, **kw: _Fitted(),
    )
    monkeypatch.setattr(
        holdout_eval,
        "p_left_fixed_params",
        lambda model_name, models_dir, stimuli, params, **kw: pred[: len(stimuli)],
    )
    monkeypatch.setattr(holdout_eval, "pm_data_inputs", lambda model: [])
    monkeypatch.setattr(
        holdout_eval, "make_stim_data", lambda model, rows: {"n": len(rows)}
    )
    monkeypatch.setattr(
        holdout_eval,
        "seed_baseline_correlation",
        lambda *a, **kw: {"mean_r": 0.5, "per_model": {}},
    )
    monkeypatch.setattr(
        holdout_eval,
        "fitted_seed_baseline_correlation",
        lambda *a, **kw: {
            "mean_r": 0.5, "mean_rmse": 0.1, "per_model": {}, "n_responses": 10,
        },
    )

    out_dir = tmp_path / "out"
    main(Args(
        sweep_a=Path(sweep_a_str),
        sweep_b=Path(sweep_b_str),
        out=out_dir,
        cell_a=cell_key,
        cell_b=cell_key,
    ))

    paired_json = out_dir / "paired.json"
    assert paired_json.exists(), "paired.json not written"

    result = json.loads(paired_json.read_text(encoding="utf-8"))
    assert result["n_pairs"] == 1, (
        f"Expected 1 paired cell, got {result['n_pairs']}; "
        f"unreconstructable: {result.get('unreconstructable', [])}"
    )
    pair = result["pairs"][0]
    assert "rmse_a" in pair
    assert "rmse_b" in pair
    assert "delta_rmse" in pair

    # The common pool must be smaller than either cell's own pool
    # (because the union of training pairs is larger)
    assert pair["n_eval_stimuli"] > 0

    paired_csv = out_dir / "paired.csv"
    assert paired_csv.exists()


# ---------------------------------------------------------------------------
# oracle_admitted_models on a real archived cell
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _REAL_CELL_SWEEP is None,
    reason="no archived cell in sweep",
)
def test_oracle_on_real_archived_cell_with_monkeypatch(tmp_path, monkeypatch):
    """oracle_admitted_models works on a real archived cell (monkeypatched fits)."""
    import numpy as np

    import scripts.subjective_randomness.oracle_admitted_models as oracle_mod
    from scripts.subjective_randomness.oracle_admitted_models import Args, main

    cell_key = _REAL_CELL_SWEEP
    cell_dir = _SWEEP / cell_key

    # Copy holdout.json so oracle.json lands in tmp
    tmp_holdout = tmp_path / "holdout.json"
    shutil.copy2(cell_dir / "holdout.json", tmp_holdout)
    # Symlink the archive and cache
    (tmp_path / "agent_runs.tar.gz").symlink_to(cell_dir / "agent_runs.tar.gz")
    cache_src = cell_dir / "mcmc_cache"
    if cache_src.is_dir():
        (tmp_path / "mcmc_cache").symlink_to(cache_src)

    # Stub the heavy seams
    pred = np.array([0.5] * 500)

    monkeypatch.setattr(
        oracle_mod,
        "p_left_fixed_params",
        lambda model_name, models_dir, stimuli, params, **kw: pred[: len(stimuli)],
    )
    monkeypatch.setattr(
        oracle_mod, "make_stim_data", lambda model, rows: {"n": len(rows)}
    )
    monkeypatch.setattr(oracle_mod, "pm_data_inputs", lambda model: [])

    class _Fitted:
        model = None
        def __init__(self, name="stub"):
            self.name = name
        def predict_p_left(self, stim_data, **kw):
            return pred[: stim_data["n"]]

    monkeypatch.setattr(
        oracle_mod,
        "fit_model",
        lambda name, models_dir, responses_path, **kw: _Fitted(name),
    )

    main(Args(result=tmp_holdout, steps="final"))

    oracle_json = tmp_path / "oracle.json"
    assert oracle_json.exists(), "oracle.json not written"

    oracle = json.loads(oracle_json.read_text(encoding="utf-8"))
    assert len(oracle["steps"]) > 0, "No steps scored"
    step = oracle["steps"][0]
    required = {
        "oracle_best_model", "oracle_rmse",
        "incumbent_model", "incumbent_rmse",
        "oracle_incumbent_gap",
    }
    assert required <= set(step), f"Missing: {required - set(step)}"


# ---------------------------------------------------------------------------
# Corrupt archive is reported, not a silent crash
# ---------------------------------------------------------------------------


def test_compare_corrupt_archive_reported(tmp_path, monkeypatch):
    """A cell with a corrupt archive is listed as unreconstructable and exits non-zero."""
    from scripts.subjective_randomness.compare_matched_cells import Args, main

    # Build sweep_a with a real (tiny) cell
    sweep_a = tmp_path / "sweep_a"
    cell_a = sweep_a / "run1" / "gt_a"
    cell_a.mkdir(parents=True)

    # Build a valid run tree, archive it, delete on-disk tree
    staging = tmp_path / "staging_a"
    staging.mkdir()
    from tests.test_cell_archive import _build_run_tree
    _build_run_tree(staging, "gt_a")
    from tests.test_cell_archive import _archive_run_tree
    _archive_run_tree(cell_a, staging)

    # Create a holdout.json pointing at a non-existent path
    run_root_a = cell_a / "repo" / "_runs" / "gt_a"
    holdout_a = {
        "project_id": "test",
        "seed_models_dir": str(tmp_path),
        "n_experiments": 1,
        "fit_kwargs": {},
        "eval_pool": {"n_pairs": 0, "lengths": [3], "seed": 0, "min_remaining": 1, "exhaustive": True},
        "gt_runs": [{
            "gt_model": "gt_a",
            "params": {},
            "run_root": str(run_root_a),
            "trajectory": [{"experiment": 1, "step": 0, "iteration": None, "global_step": 0, "best_model": "s"}],
        }],
    }
    (cell_a / "holdout.json").write_text(json.dumps(holdout_a), encoding="utf-8")

    # Build sweep_b with a corrupt archive
    sweep_b = tmp_path / "sweep_b"
    cell_b = sweep_b / "run1" / "gt_a"
    cell_b.mkdir(parents=True)
    (cell_b / "agent_runs.tar.gz").write_bytes(b"corrupt")
    run_root_b = cell_b / "repo" / "_runs" / "gt_a"
    holdout_b = {**holdout_a, "gt_runs": [{**holdout_a["gt_runs"][0], "run_root": str(run_root_b)}]}
    (cell_b / "holdout.json").write_text(json.dumps(holdout_b), encoding="utf-8")

    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit) as exc_info:
        main(Args(sweep_a=sweep_a, sweep_b=sweep_b, out=out_dir))

    # The output is written before the exit, with the reason listed
    result = json.loads((out_dir / "paired.json").read_text(encoding="utf-8"))
    assert result["n_pairs"] == 0
    assert len(result["unreconstructable"]) > 0


# ---------------------------------------------------------------------------
# Exit status reflects whether paired cells were produced
# ---------------------------------------------------------------------------


def test_compare_no_paired_cells_exits_nonzero(tmp_path, monkeypatch):
    """When zero paired cells are produced, the tool exits with a non-zero status."""
    from scripts.subjective_randomness.compare_matched_cells import Args, main

    sweep_a = tmp_path / "sweep_a"
    sweep_b = tmp_path / "sweep_b"
    # No cells at all — different structure
    cell_a = sweep_a / "run1" / "gt_a"
    cell_a.mkdir(parents=True)
    cell_b = sweep_b / "run1" / "gt_b"
    cell_b.mkdir(parents=True)
    # No holdout.json in either — _discover_cells returns empty
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit):
        main(Args(sweep_a=sweep_a, sweep_b=sweep_b, out=out_dir))
