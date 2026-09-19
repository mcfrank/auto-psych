"""Tests for the oracle_admitted_models CLI.

The oracle diagnostic scores every model the inner loop ever held and reports
the oracle-best vs. the incumbent, separating discovery, selection, and
retention failures.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

import src.subjective_randomness.holdout_recovery as holdout_recovery
from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger, LedgerEntry

import scripts.subjective_randomness.oracle_admitted_models as oracle_mod


def _write_model_file(models_dir: Path, name: str) -> None:
    """Write a trivial Python file that a model loader would find."""
    (models_dir / f"{name}.py").write_text(
        f"# stub model {name}\nimport pymc as pm\nmodel = pm.Model()\n",
        encoding="utf-8",
    )


def _build_synthetic_run(
    run_root: Path,
    n_experiments: int,
    history_by_exp: dict[int, list[dict]],
    models_by_exp: dict[int, list[str]],
    pruned_by_exp: dict[int, list[str]] | None = None,
    ledger_entries: list[LedgerEntry] | None = None,
) -> None:
    """Build a synthetic run tree with history, models, and optional ledger."""
    for exp_num in range(1, n_experiments + 1):
        loop_dir = run_root / f"experiment{exp_num}" / "model_loop"
        models_dir = loop_dir / "models"
        models_dir.mkdir(parents=True)
        (loop_dir / "responses.csv").write_text("chose_left\n1\n", encoding="utf-8")
        (loop_dir / "history.json").write_text(
            json.dumps(history_by_exp[exp_num]), encoding="utf-8"
        )
        data_dir = run_root / f"experiment{exp_num}" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "responses.csv").write_text(
            "sequence_a,sequence_b,chose_left\nHHH,TTT,1\n", encoding="utf-8"
        )
        for name in models_by_exp.get(exp_num, []):
            _write_model_file(models_dir, name)
        if pruned_by_exp and exp_num in pruned_by_exp:
            pruned_dir = models_dir / "pruned"
            pruned_dir.mkdir(exist_ok=True)
            for name in pruned_by_exp[exp_num]:
                _write_model_file(pruned_dir, name)

    if ledger_entries:
        ledger_path = run_root / f"experiment{n_experiments}" / "model_loop" / "attempted_hypotheses.jsonl"
        ledger = HypothesisLedger.create(ledger_path, inherit_from=None)
        for entry in ledger_entries:
            ledger.append(entry)


def _patch_oracle_seams(monkeypatch, gt_p, predictions):
    """Patch the oracle module's imported evaluation functions."""
    monkeypatch.setattr(
        oracle_mod,
        "p_left_fixed_params",
        lambda model_name, models_dir, stimuli, params, **kw: gt_p[:len(stimuli)],
    )
    monkeypatch.setattr(
        oracle_mod, "make_stim_data", lambda model, rows: {"n": len(rows)}
    )
    monkeypatch.setattr(oracle_mod, "pm_data_inputs", lambda model: [])

    class Fitted:
        model = None
        def __init__(self, name):
            self.name = name
        def predict_p_left(self, stim_data):
            return predictions[self.name][:stim_data["n"]]

    monkeypatch.setattr(
        oracle_mod,
        "fit_model",
        lambda name, models_dir, responses_path, **kw: Fitted(name),
    )


def test_oracle_admitted_models_reports_oracle_gap(tmp_path, monkeypatch):
    """Oracle diagnostic shows the oracle-best vs. incumbent gap."""
    from scripts.subjective_randomness.oracle_admitted_models import Args, main

    run_root = tmp_path / "gt_a"

    history = {
        1: [
            {
                "step": 0, "iteration": None,
                "best_model": "seed_a",
                "posteriors": {"seed_a": 0.5, "seed_b": 0.3, "candidate_1": 0.2},
                "elpd_loo": {"seed_a": -1.0, "seed_b": -2.0, "candidate_1": -0.5},
            },
        ]
    }
    models = {1: ["seed_a", "seed_b", "candidate_1"]}
    pruned = {1: ["loser"]}

    ledger_entries = [
        LedgerEntry("seed_a", "admitted", "seed", "hypothesis A", "experiment1"),
        LedgerEntry("seed_b", "admitted", "seed", "hypothesis B", "experiment1"),
        LedgerEntry("candidate_1", "admitted", "ok", "hypothesis C", "experiment1"),
        LedgerEntry("loser", "pruned", "behind by 3 dse", "hypothesis D", "experiment1"),
    ]

    _build_synthetic_run(run_root, 1, history, models, pruned, ledger_entries)

    gt_p = np.array([0.3, 0.5, 0.7])
    predictions = {
        "seed_a": np.array([0.35, 0.55, 0.65]),
        "seed_b": np.array([0.4, 0.6, 0.5]),
        "candidate_1": np.array([0.31, 0.51, 0.69]),
        "loser": np.array([0.29, 0.49, 0.71]),
    }

    _patch_oracle_seams(monkeypatch, gt_p, predictions)

    holdout_result = {
        "n_experiments": 1,
        "seed_models_dir": str(tmp_path),
        "fit_kwargs": {},
        "eval_pool": {
            "n_pairs": 0, "lengths": [3], "seed": 0,
            "min_remaining": 1, "exhaustive": True, "predict_max_draws": None,
        },
        "gt_runs": [{
            "gt_model": "gt_a",
            "params": {"theta_alt": 0.65},
            "run_root": str(run_root),
            "trajectory": history[1],
        }],
    }
    holdout_json = tmp_path / "holdout.json"
    holdout_json.write_text(json.dumps(holdout_result, indent=2), encoding="utf-8")

    eval_stimuli = [
        {"sequence_a": "HHT", "sequence_b": "TTH"},
        {"sequence_a": "HTH", "sequence_b": "THT"},
        {"sequence_a": "HHH", "sequence_b": "TTT"},
    ]
    (run_root / "eval_stimuli.json").write_text(json.dumps(eval_stimuli), encoding="utf-8")

    main(Args(result=holdout_json))

    oracle_json = tmp_path / "oracle.json"
    assert oracle_json.exists()

    oracle = json.loads(oracle_json.read_text(encoding="utf-8"))
    assert "steps" in oracle
    assert len(oracle["steps"]) >= 1

    step = oracle["steps"][0]
    assert "oracle_best_model" in step
    assert "oracle_rmse" in step
    assert "incumbent_model" in step
    assert "incumbent_rmse" in step
    assert "oracle_incumbent_gap" in step

    assert step["oracle_rmse"] <= step["incumbent_rmse"]

    oracle_csv = tmp_path / "oracle.csv"
    assert oracle_csv.exists()
    with oracle_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 1
    assert "oracle_best_model" in rows[0]


def test_oracle_extracts_from_archive(tmp_path, monkeypatch):
    """When run_root doesn't exist on disk, oracle extracts from agent_runs.tar.gz."""
    import tarfile

    from scripts.subjective_randomness.oracle_admitted_models import Args, main

    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()

    # Build a run tree under a temp location, then archive and remove it.
    staging = tmp_path / "staging"
    run_root_rel = Path("_runs") / "gt_a"
    run_root_abs = staging / run_root_rel
    run_root_abs.mkdir(parents=True)

    history = {
        1: [
            {
                "step": 0, "iteration": None,
                "best_model": "seed_a",
                "posteriors": {"seed_a": 0.5, "seed_b": 0.5},
                "elpd_loo": {"seed_a": -1.0, "seed_b": -2.0},
            },
        ]
    }
    models = {1: ["seed_a", "seed_b"]}
    _build_synthetic_run(run_root_abs, 1, history, models)

    eval_stimuli = [
        {"sequence_a": "HHT", "sequence_b": "TTH"},
        {"sequence_a": "HTH", "sequence_b": "THT"},
        {"sequence_a": "HHH", "sequence_b": "TTT"},
    ]
    (run_root_abs / "eval_stimuli.json").write_text(
        json.dumps(eval_stimuli), encoding="utf-8"
    )

    # Archive into cell_dir/agent_runs.tar.gz
    tar_path = cell_dir / "agent_runs.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(str(staging / "_runs"), arcname="_runs")

    # The holdout.json run_root points to a non-existent path
    fake_run_root = cell_dir / "repo" / "_runs" / "gt_a"

    gt_p = np.array([0.3, 0.5, 0.7])
    predictions = {
        "seed_a": np.array([0.35, 0.55, 0.65]),
        "seed_b": np.array([0.4, 0.6, 0.5]),
    }
    _patch_oracle_seams(monkeypatch, gt_p, predictions)

    holdout_result = {
        "n_experiments": 1,
        "seed_models_dir": str(tmp_path),
        "fit_kwargs": {},
        "eval_pool": {
            "n_pairs": 0, "lengths": [3], "seed": 0,
            "min_remaining": 1, "exhaustive": True, "predict_max_draws": None,
        },
        "gt_runs": [{
            "gt_model": "gt_a",
            "params": {"theta_alt": 0.65},
            "run_root": str(fake_run_root),
            "trajectory": history[1],
        }],
    }
    holdout_json = cell_dir / "holdout.json"
    holdout_json.write_text(json.dumps(holdout_result, indent=2), encoding="utf-8")

    main(Args(result=holdout_json))

    oracle_json = cell_dir / "oracle.json"
    assert oracle_json.exists()

    oracle = json.loads(oracle_json.read_text(encoding="utf-8"))
    assert len(oracle["steps"]) >= 1
    step = oracle["steps"][0]
    assert step["oracle_rmse"] <= step["incumbent_rmse"]


def test_oracle_reports_lost_incumbents(tmp_path, monkeypatch):
    """A model that was best_model at an earlier step but was later pruned."""
    from scripts.subjective_randomness.oracle_admitted_models import Args, main

    run_root = tmp_path / "gt_a"

    history = {
        1: [
            {
                "step": 0, "iteration": None,
                "best_model": "former_champ",
                "posteriors": {"former_champ": 1.0},
                "elpd_loo": {"former_champ": -1.0},
            },
            {
                "step": 1, "iteration": 0,
                "best_model": "new_winner",
                "posteriors": {"new_winner": 0.8, "former_champ": 0.2},
                "elpd_loo": {"new_winner": -0.5, "former_champ": -3.0},
                "pruned": ["former_champ"],
            },
        ]
    }
    models = {1: ["new_winner"]}
    pruned = {1: ["former_champ"]}

    ledger_entries = [
        LedgerEntry("former_champ", "admitted", "seed", "hypothesis A", "experiment1"),
        LedgerEntry("new_winner", "admitted", "ok", "hypothesis B", "experiment1 round 0"),
        LedgerEntry("former_champ", "pruned", "behind by 4 dse", "hypothesis A", "experiment1 step 1"),
    ]

    _build_synthetic_run(run_root, 1, history, models, pruned, ledger_entries)

    gt_p = np.array([0.3, 0.5, 0.7])
    predictions = {
        "former_champ": np.array([0.32, 0.48, 0.68]),
        "new_winner": np.array([0.35, 0.55, 0.65]),
    }

    _patch_oracle_seams(monkeypatch, gt_p, predictions)

    holdout_result = {
        "n_experiments": 1,
        "seed_models_dir": str(tmp_path),
        "fit_kwargs": {},
        "eval_pool": {
            "n_pairs": 0, "lengths": [3], "seed": 0,
            "min_remaining": 1, "exhaustive": True, "predict_max_draws": None,
        },
        "gt_runs": [{
            "gt_model": "gt_a",
            "params": {"theta_alt": 0.65},
            "run_root": str(run_root),
            "trajectory": history[1],
        }],
    }
    holdout_json = tmp_path / "holdout.json"
    holdout_json.write_text(json.dumps(holdout_result, indent=2), encoding="utf-8")

    eval_stimuli = [
        {"sequence_a": "HHT", "sequence_b": "TTH"},
        {"sequence_a": "HTH", "sequence_b": "THT"},
        {"sequence_a": "HHH", "sequence_b": "TTT"},
    ]
    (run_root / "eval_stimuli.json").write_text(json.dumps(eval_stimuli), encoding="utf-8")

    main(Args(result=holdout_json, steps="all"))

    oracle = json.loads((tmp_path / "oracle.json").read_text(encoding="utf-8"))
    lost = oracle.get("lost_incumbents", [])
    assert any(l["model"] == "former_champ" for l in lost)


# ---------------------------------------------------------------------------
# Integration test against a real archived cell from the sweep
# ---------------------------------------------------------------------------

_SWEEP_ROOT = Path(
    os.environ.get(
        "SWEEP_ROOT",
        "/scratch/users/benpry/auto-psych/consolidation_2026_09/sweep",
    )
)
_REAL_CELL = "run1/falk_konold_dp"


def _real_cell_exists() -> bool:
    cell_dir = _SWEEP_ROOT / _REAL_CELL
    return (cell_dir / "holdout.json").exists() and (
        (cell_dir / "agent_runs.tar.gz").exists()
    )


@pytest.mark.slow
@pytest.mark.skipif(not _real_cell_exists(), reason="sweep cell not available")
def test_oracle_against_real_archived_cell(tmp_path):
    """Run oracle on an actual archived cell (cached fits, ``--steps final``).

    This exercises the archive extraction and cache-dir resolution against
    a real cell, catching the regressions that sank P15 twice.
    """
    import shutil
    import subprocess

    from pyprojroot import here

    cell_dir = _SWEEP_ROOT / _REAL_CELL

    # Copy holdout.json into tmp_path so oracle.json lands there.
    tmp_holdout = tmp_path / "holdout.json"
    shutil.copy2(cell_dir / "holdout.json", tmp_holdout)

    # Symlink the archive and mcmc_cache.
    (tmp_path / "agent_runs.tar.gz").symlink_to(cell_dir / "agent_runs.tar.gz")
    cache_src = cell_dir / "mcmc_cache"
    if cache_src.is_dir():
        (tmp_path / "mcmc_cache").symlink_to(cache_src)

    venv_py = os.environ.get(
        "VENV_PY",
        "/scratch/users/benpry/auto-psych/consolidation_2026_09/venv/bin/python",
    )
    result = subprocess.run(
        [
            venv_py,
            "scripts/subjective_randomness/oracle_admitted_models.py",
            "--result", str(tmp_holdout),
            "--steps", "final",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(here()),
    )
    assert result.returncode == 0, (
        f"oracle_admitted_models.py failed:\n"
        f"stdout: {result.stdout[-1000:]}\nstderr: {result.stderr[-1000:]}"
    )

    oracle_json = tmp_path / "oracle.json"
    assert oracle_json.exists(), "oracle.json not written"
    oracle = json.loads(oracle_json.read_text(encoding="utf-8"))

    assert "steps" in oracle
    assert "lost_incumbents" in oracle
    assert len(oracle["steps"]) > 0

    step = oracle["steps"][0]
    required = {
        "gt_model", "experiment", "step", "n_models_scored",
        "oracle_best_model", "oracle_rmse",
        "incumbent_model", "incumbent_rmse",
        "oracle_incumbent_gap", "final_model", "final_rmse",
    }
    assert required <= set(step), f"Missing fields: {required - set(step)}"
    assert isinstance(step["oracle_rmse"], (int, float))
    assert step["oracle_rmse"] >= 0
    assert step["n_models_scored"] >= 1

    oracle_csv = tmp_path / "oracle.csv"
    assert oracle_csv.exists(), "oracle.csv not written"
