"""The live model-loop stage caches its fits, so no fit is sampled twice.

The admission gates reuse a model's fit only through the on-disk fit cache: the
real-fit gate fits the candidate, and the novelty gate asks ``fit_model`` for
the posterior of the candidate and of every admitted model, "from the cached
fits". ``run.py``'s model-loop stage passed the inner loop no ``cache_dir``, so
on the live path (and the simulated one, which runs the same stage) every
novelty check re-sampled every admitted model and the concurrent prefit of a
wave never ran. The holdout harness always passed one. In run 1 of the full
live run (2026-09-29) round 0 took about 4.5 hours: one model was sampled 14
times on the same data (148 minutes), and each round grows with the set.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

import src.pipelines.outer_loop.run as outer_run
from src.models import pymc_inference as pi
from src.pipelines.inner_loop import model_zoo, pymc_orchestrator
from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger
from src.pipelines.outer_loop import model_loop_runner as mlr
from tests.paths import PYMC_MODEL_FIXTURES_DIR

PROJECT = "subjective_randomness"


def test_the_live_model_stage_gives_the_loop_a_fit_cache_inside_model_loop(
    tmp_path, monkeypatch
):
    seen = {}
    monkeypatch.setattr(outer_run, "begin_model_loop_stage", lambda exp_dir: None)
    monkeypatch.setattr(outer_run, "finish_model_loop_stage", lambda exp_dir: None)
    monkeypatch.setattr(
        outer_run, "run_inner_model_loop_programmatic", lambda *a, **k: seen.update(k)
    )
    outer_run._run_agent(
        "5_model_loop", tmp_path, PROJECT, 1, "live", 40, None, False,
    )
    # Inside model_loop/, which every (re)start of the stage empties, and named
    # .fit_cache, which results collection leaves out (hundreds of MB a fit).
    assert seen["cache_dir"] == tmp_path / "model_loop" / ".fit_cache"


def _experiment_one(tmp_path: Path) -> Path:
    """Experiment 1 of a run as the live stage finds it: collected responses
    and a seeded set (one fixture model, an empty ledger), no model loop yet."""
    exp_dir = tmp_path / "output" / PROJECT / "experiment1"
    (exp_dir / "data").mkdir(parents=True)
    # The fixture responses, with the integer participant ids the novelty gate
    # marginalizes over (p0 -> 0).
    (exp_dir / "data" / "responses.csv").write_text(
        re.sub(
            r",p(\d+),",
            r",\1,",
            (PYMC_MODEL_FIXTURES_DIR / "responses.csv").read_text(),
        ),
        encoding="utf-8",
    )
    models = exp_dir / "cognitive_models"
    models.mkdir()
    (models / "bayesian_fair_coin.py").write_text(
        (PYMC_MODEL_FIXTURES_DIR / "bayesian_fair_coin.py").read_text(),
        encoding="utf-8",
    )
    (models / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": "bayesian_fair_coin", "rationale": "the seed"}]}
        ),
        encoding="utf-8",
    )
    (models / "attempted_hypotheses.jsonl").write_text("", encoding="utf-8")
    mlr.init_registry(exp_dir)
    mlr.agent_notes_dir(exp_dir).mkdir()
    return exp_dir


@pytest.mark.slow
def test_the_live_model_stage_samples_no_fit_twice(tmp_path, monkeypatch):
    exp_dir = _experiment_one(tmp_path)

    # The candidate agent writes a genuinely different model (it must be
    # admitted, so the novelty gate compares it with the seed).
    def write_candidate(candidate_dir, docs, **kwargs):
        (candidate_dir / "candidate.py").write_text(
            (PYMC_MODEL_FIXTURES_DIR / "representativeness.py").read_text(),
            encoding="utf-8",
        )
        (candidate_dir / "hypothesis.md").write_text(
            "People prefer the sequence whose share of heads is nearer one half.\n",
            encoding="utf-8",
        )
        (candidate_dir / "model_name.txt").write_text(
            "head_share", encoding="utf-8"
        )
        return True

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", write_candidate)

    # Thirty fixture trials give a stochastic Pareto-k warning; the export and
    # pruning need a trusted fit, so every row is read as reliable (as in
    # test_pymc_inner_loop). The fits behind the table are the real ones.
    def reliable(compare):
        def compare_reliably(*args, **kwargs):
            return {
                name: {**row, "loo_unreliable": False}
                for name, row in compare(*args, **kwargs).items()
            }

        return compare_reliably

    monkeypatch.setattr(
        pymc_orchestrator, "_compare", reliable(pymc_orchestrator._compare)
    )
    monkeypatch.setattr(model_zoo, "compare_table", reliable(model_zoo.compare_table))

    # Every sampling run in this (the harness) process, by fit fingerprint. A
    # fit a child process sampled (a pool worker, a time-limited candidate fit)
    # is loaded here from a file and is not a sampling run of this process.
    sampled = []
    real_fit_once = pi._fit_once

    def recording_fit_once(name, models_dir, responses_path, settings, cache_dir):
        fingerprint = pi.fit_fingerprint(name, models_dir, responses_path, settings)
        loads = (
            cache_dir is not None
            and pi.cached_fit_path(cache_dir, name, fingerprint).exists()
        )
        fitted = real_fit_once(name, models_dir, responses_path, settings, cache_dir)
        if not loads:
            sampled.append((name, fingerprint))
        return fitted

    monkeypatch.setattr(pi, "_fit_once", recording_fit_once)
    pi.clear_fit_cache()
    try:
        outer_run._run_agent(
            "5_model_loop",
            exp_dir,
            PROJECT,
            1,
            "simulated_participants_nobrowser",
            4,
            None,
            False,
            inner_loop_iterations=1,
            inner_loop_candidates=1,
            fit_kwargs={
                "draws": 500,
                "tune": 500,
                "chains": 2,
                "cores": 2,
                "target_accept": 0.8,
            },
            enable_critique=False,
        )
    finally:
        pi.clear_fit_cache()

    # Admitted, so the novelty gate compared it with the seed (the fair-coin
    # data then prune it at the end of the experiment).
    ledger = HypothesisLedger(exp_dir / "model_loop" / "attempted_hypotheses.jsonl")
    assert ("head_share", "admitted") in [(e.name, e.outcome) for e in ledger.entries()]
    assert [fit for fit, n in Counter(sampled).items() if n > 1] == []
    fit_cache = exp_dir / "model_loop" / ".fit_cache"
    cached = {p.name.split(".")[0] for p in fit_cache.glob("*.nc")}
    assert cached >= {"bayesian_fair_coin", "head_share"}
