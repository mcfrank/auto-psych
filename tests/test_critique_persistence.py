"""The inner-loop orchestrator must always persist critique results.

The critique agent only proposes test statistics; the orchestrator runs the
posterior-predictive harness itself and writes ``ppc_results.json`` + a derived
``critiques.md`` so the critique results are recorded deterministically (not
left to whether the agent happened to run the harness).
"""

from __future__ import annotations

import json
from pathlib import Path

import src.critique.ppc as ppc
from src.pipelines.inner_loop.pymc_orchestrator import (
    _format_critiques_md,
    _persist_critique_results,
)

_RESULT = {
    "model": "bayesian_fair_coin",
    "n_test_statistics": 2,
    "n_replicates": 200,
    "significance_alpha": 0.05,
    "n_significant": 1,
    "results": [
        {"name": "alternation_gap", "description": "alternation proportion of A",
         "t_observed": 0.55, "null_mean": 0.50, "null_std": 0.02, "z_score": 2.5,
         "p_value": 0.012, "p_value_adjusted": 0.024, "significant": True, "error": None},
        {"name": "max_run", "description": "max run length",
         "t_observed": 2.1, "null_mean": 2.0, "null_std": 0.3, "z_score": 0.5,
         "p_value": 0.6, "p_value_adjusted": 0.6, "significant": False, "error": None},
    ],
}


def test_format_critiques_md_lists_significant_discrepancies():
    md = _format_critiques_md(_RESULT)
    assert "1 of 2" in md
    assert "Significant discrepancies" in md
    assert "alternation_gap" in md
    assert "max_run" not in md  # not significant -> not listed


def test_persist_skips_when_no_test_statistics(tmp_path: Path, monkeypatch):
    called = False

    def _fail(*a, **k):
        nonlocal called
        called = True
        raise AssertionError("PPC should not run without test statistics")

    monkeypatch.setattr(ppc, "run_ppc_for_model", _fail)
    crit = tmp_path / "critique"
    crit.mkdir()
    _persist_critique_results(
        crit, "m", models_dir=tmp_path, responses_path=tmp_path / "r.csv",
        fit_cache_dir=tmp_path, fit_kwargs={}, n_replicates=10, significance_alpha=0.05,
    )
    assert not called
    assert not (crit / "ppc_results.json").exists()


def test_persist_writes_results_and_critiques(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ppc, "run_ppc_for_model", lambda *a, **k: _RESULT)
    crit = tmp_path / "critique"
    (crit / "test_stats").mkdir(parents=True)
    (crit / "test_stats" / "alternation_gap.py").write_text(
        "def test_statistic(df):\n    return 0.0\n"
    )

    _persist_critique_results(
        crit, "bayesian_fair_coin", models_dir=tmp_path,
        responses_path=tmp_path / "r.csv", fit_cache_dir=tmp_path,
        fit_kwargs={}, n_replicates=200, significance_alpha=0.05,
    )

    saved = json.loads((crit / "ppc_results.json").read_text())
    assert saved["n_significant"] == 1
    assert {r["name"] for r in saved["results"]} == {"alternation_gap", "max_run"}
    assert "alternation_gap" in (crit / "critiques.md").read_text()
