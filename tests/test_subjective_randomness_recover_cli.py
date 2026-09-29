"""CLI tests for the closed-ended model-recovery script.

These pin the argument layer of `scripts/subjective_randomness/model_recovery.py`
(the script lives under `scripts/`, not as an importable package) so the
optional `--tidy-csv` output flag cannot silently regress. The recovery
business logic is tested elsewhere.
"""

from __future__ import annotations

import csv
from pathlib import Path

import tyro
from tests.paths import REPO_ROOT, load_script_module

SCRIPTS = REPO_ROOT / "scripts" / "subjective_randomness"

# A canned runner output so the --tidy-csv branch can be exercised end-to-end
# without running MCMC (the expensive runner is monkeypatched).
CONFUSION_RESULT = {
    "seed_models": ["A", "B"],
    "generator": "pymc",
    "n_participants": 5,
    "n_stimuli": 4,
    "fit_kwargs": {},
    "generating": [
        {
            "generating_model": "A",
            "params": {},
            "best_model": "A",
            "recovered_correct": True,
            "posteriors": {"A": 0.8, "B": 0.2},
            "elpd_loo": {"A": -1.0, "B": -2.0},
        }
    ],
}
CONFUSION_TIDY_COLUMNS = {
    "generating_model",
    "recovered_model",
    "posterior",
    "elpd_loo",
    "is_true_model",
    "is_best_model",
}


def _load_script(name: str):
    return load_script_module(SCRIPTS / name, f"_sr_script_{name[:-3]}")


def test_model_recovery_cli_defaults_and_overrides():
    args_cls = _load_script("model_recovery.py").Args

    default = tyro.cli(args_cls, args=["--config", "c.yaml", "--out", "r.json"])
    assert default.config == Path("c.yaml")
    assert default.out == Path("r.json")
    assert default.tidy_csv is None
    assert default.results_root is None
    assert default.n_participants is None
    assert default.draws is None  # falls back to the config's fit settings
    assert default.generator is None  # falls back to the config's generator

    full = tyro.cli(
        args_cls,
        args=[
            "--config", "c.yaml",
            "--out", "r.json",
            "--tidy-csv", "c.csv",
            "--n-participants", "12",
            "--draws", "300",
            "--chains", "4",
            "--generator", "model_family",
        ],
    )
    assert full.tidy_csv == Path("c.csv")
    assert full.n_participants == 12
    assert full.draws == 300
    assert full.chains == 4
    assert full.generator == "model_family"


# ── --tidy-csv branch executes (would fail if the branch were reverted) ──


def _write_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text("model: demo\n", encoding="utf-8")
    return config


def test_model_recovery_writes_tidy_csv_when_flag_set(tmp_path, monkeypatch):
    mod = _load_script("model_recovery.py")
    monkeypatch.setattr(mod, "run_recovery_from_config", lambda *a, **k: CONFUSION_RESULT)
    tidy = tmp_path / "confusion.csv"

    mod.main(
        mod.Args(config=_write_config(tmp_path), out=tmp_path / "c.json", tidy_csv=tidy)
    )

    assert tidy.exists()
    rows = list(csv.DictReader(tidy.open(encoding="utf-8")))
    assert set(rows[0]) == CONFUSION_TIDY_COLUMNS
    assert len(rows) == 2  # 1 generating model x 2 recovered (seed) models
