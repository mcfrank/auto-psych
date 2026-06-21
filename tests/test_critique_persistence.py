"""The inner-loop orchestrator must always persist critique results.

The critique agent only proposes test statistics; the orchestrator runs the
posterior-predictive harness itself and writes ``ppc_results.json`` + a derived
``critiques.md`` so the critique results are recorded deterministically (not
left to whether the agent happened to run the harness).

The critique must also never silently produce *nothing*: if the agent proposes
no statistics (e.g. it could not locate its context), a deterministic default
battery is written so the posterior-predictive check still runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import src.critique.ppc as ppc
from src.pipelines.inner_loop import pymc_orchestrator
from src.pipelines.inner_loop.pymc_orchestrator import (
    _format_critiques_md,
    _persist_critique_results,
    _write_default_test_statistics,
)
from src.runtime.config import REPO_ROOT

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


_RESPONSES_CSV = (
    "chose_left,p_alts_a,imbalance_a\n"
    "1,0.50,0.10\n0,0.30,0.20\n1,0.70,0.00\n0,0.40,0.30\n1,0.60,0.05\n"
)


def test_persist_uses_default_battery_when_agent_writes_no_stats(tmp_path: Path, monkeypatch):
    """With no agent statistics, the critique must NOT skip: it writes a default
    battery and still runs the PPC harness over it."""
    responses = tmp_path / "responses.csv"
    responses.write_text(_RESPONSES_CSV, encoding="utf-8")
    # Avoid loading a real PyMC model just to learn the response column.
    monkeypatch.setattr(
        pymc_orchestrator, "_incumbent_response_col", lambda *a, **k: "chose_left"
    )

    seen = {}

    def fake_ppc(model, models_dir, responses_path, test_stats_dir, **k):
        seen["n_files"] = len(sorted(Path(test_stats_dir).glob("*.py")))
        return {
            "model": model,
            "n_test_statistics": seen["n_files"],
            "n_replicates": k.get("n_replicates"),
            "significance_alpha": k.get("significance_alpha"),
            "n_significant": 0,
            "results": [],
        }

    monkeypatch.setattr(ppc, "run_ppc_for_model", fake_ppc)
    crit = tmp_path / "critique"
    crit.mkdir()
    _persist_critique_results(
        crit, "m", models_dir=tmp_path, responses_path=responses,
        fit_cache_dir=tmp_path, fit_kwargs={}, n_replicates=10, significance_alpha=0.05,
    )
    assert seen["n_files"] > 0  # default battery was written and handed to PPC
    assert (crit / "ppc_results.json").exists()
    assert (crit / "critiques.md").exists()


def test_default_test_statistics_are_valid_loadable_functions(tmp_path: Path):
    """The generated fallback statistics must load and evaluate to finite floats."""
    import pandas as pd

    responses = tmp_path / "responses.csv"
    responses.write_text(_RESPONSES_CSV, encoding="utf-8")
    stats_dir = tmp_path / "test_stats"
    n = _write_default_test_statistics(stats_dir, responses, "chose_left")

    assert n >= 2  # the marginal-response stat + at least one feature correlation
    df = pd.read_csv(responses)
    files = sorted(stats_dir.glob("*.py"))
    assert len(files) == n
    for f in files:
        ts = ppc.load_test_statistic_file(f)  # must parse with name/description headers
        value = float(ppc._compile_test_statistic(ts.code)(df.copy()))
        assert math.isfinite(value)


def test_critique_prompt_names_the_critique_dir(tmp_path: Path, monkeypatch):
    """The critique agent runs from REPO_ROOT, so its prompt must name the critique
    dir explicitly — otherwise it cannot reliably locate CRITIQUE_CONTEXT.md."""
    import src.runtime.coding_agent as coding_agent

    monkeypatch.setattr(pymc_orchestrator, "_seed_critique_fit_cache", lambda *a, **k: None)
    monkeypatch.setattr(pymc_orchestrator, "_write_critique_context", lambda *a, **k: None)
    monkeypatch.setattr(pymc_orchestrator, "_persist_critique_results", lambda *a, **k: None)

    captured = {}

    def fake_run(prompt, *, cwd, log_path, allowed_dirs, timeout_secs, backend):
        captured["prompt"] = prompt
        captured["cwd"] = cwd
        return True, ""

    monkeypatch.setattr(coding_agent, "run_coding_agent", fake_run)
    crit = tmp_path / "critique"
    pymc_orchestrator._spawn_critique_agent(
        crit, "incumbent", models_dir=tmp_path, responses_path=tmp_path / "r.csv",
        cache_dir=None, fit_kwargs={}, n_proposals=8, significance_alpha=0.05,
        n_replicates=10, agent_timeout_sec=10, backend="opencode",
    )
    assert str(crit) in captured["prompt"]
    assert captured["cwd"] == REPO_ROOT


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
