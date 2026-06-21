"""Builder for a tiny but structurally faithful data tree of runs.

The tree exercises every run shape the viewer must handle, at varying depths:

- ``outer_loop/demo``                       a multi-experiment run (experiment1 + smoke)
- ``recovery/holdout_runs/cond_a``          a deeply-nested full experiment
- ``recovery/confusion_runs/model_x``       a bare model-loop run
- ``thinkaloud``                            a run whose experiment has an
                                            experiment-level critique and no model loop
"""

from __future__ import annotations

import json
from pathlib import Path

# A 1x1 transparent PNG — stands in for an analysis figure on disk.
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000154a24f9f0000000049454e44ae426082"
)

# A terminal transcript with ANSI codes, exactly like the agent logs on disk.
ANSI_TRANSCRIPT = "\x1b[0m> build · gemini-3.1-pro-preview\x1b[0m\n\x1b[0m$ \x1b[0mls -la\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _full_experiment(exp: Path, *, with_critique_results: bool) -> None:
    """A complete experiment dir: theory, design, implement, data, model loop, critique."""
    cm = exp / "cognitive_models"
    # The manifest as it looks AFTER the inner loop ran: the two theory-step seeds
    # plus the exported inner-loop winner (appended by _export_inner_loop_model).
    _write(
        cm / "models_manifest.yaml",
        "models:\n"
        "- name: equally_likely\n"
        "  rationale: People judge randomness by closeness to 50% heads.\n"
        "- name: bayesian_fair_coin\n"
        "  rationale: Observers compare sequences via a log Bayes factor.\n"
        "- name: inner_loop_model\n"
        "  rationale: Best PyMC model found by the inner model-improvement loop.\n",
    )
    _write(
        cm / "theory_report.md",
        "# Theory Report\n\n## bayesian_fair_coin\n"
        "**Hypothesis:** Observers compare two sequences via the log Bayes factor.\n",
    )
    _write(cm / "equally_likely.py", "def choice_prob():\n    return 0.5\n")
    _write(cm / "bayesian_fair_coin.py", "def choice_prob():\n    return 0.7\n")
    _write(cm / "inner_loop_model.py", "def choice_prob():\n    return 0.7\n")

    design = exp / "design"
    _write(design / "design_rationale.md", "# Design Rationale\n\n**Number of stimuli**: 2 pairs\n")
    _write(design / "stimuli.json", json.dumps([
        {"sequence_a": "HT", "sequence_b": "TH"},
        {"sequence_a": "HHTT", "sequence_b": "HTHT"}]))
    _write(design / "candidates.json", json.dumps([{"sequence_a": "HT", "sequence_b": "TH"}] * 5))

    experiment = exp / "experiment"
    _write(experiment / "config.json", json.dumps(
        {"experiment_id": "demo_experiment1", "run_mode": "deployed",
         "experiment_url": "https://example.web.app"}))
    _write(experiment / "index.html", "<!doctype html><title>demo</title>")
    _write(experiment / "stimuli.json", (design / "stimuli.json").read_text())

    _write(exp / "data" / "responses.csv",
           "participant_id,trial_index,sequence_a,sequence_b,chose_left,chose_right\n"
           "0,3,HT,TH,1,0\n0,5,HHTT,HTHT,0,1\n1,3,HT,TH,1,0\n1,5,HHTT,HTHT,1,0\n")
    _write(exp / "logs" / "1_theory.jsonl", ANSI_TRANSCRIPT)

    ml = exp / "model_loop"
    _write(ml / "history.json", json.dumps([
        {"step": 0, "iteration": None, "best_model": "bayesian_fair_coin",
         "posteriors": {"equally_likely": 0.1, "bayesian_fair_coin": 0.9},
         "elpd_loo": {"equally_likely": -111.6, "bayesian_fair_coin": -108.3}},
        {"step": 1, "iteration": 0, "best_model": "bayesian_fair_coin",
         "posteriors": {"equally_likely": 0.08, "bayesian_fair_coin": 0.82, "iter0_candidate0": 0.1},
         "elpd_loo": {"equally_likely": -111.6, "bayesian_fair_coin": -108.3, "iter0_candidate0": -111.4}}]))
    _write(ml / "model_posterior.json", json.dumps({
        "posteriors": {"equally_likely": 0.08, "bayesian_fair_coin": 0.82, "iter0_candidate0": 0.1},
        "elpd_loo": {"equally_likely": -111.6, "bayesian_fair_coin": -108.3, "iter0_candidate0": -111.4},
        "n_trials": 4}))
    _write(ml / "report.md", "# Inner Model Loop Report\n\nBest model: bayesian_fair_coin\n")
    _write(ml / "best_model.py", "def choice_prob():\n    return 0.7\n")

    cand = ml / "iter_0" / "candidate_0"
    _write(cand / "hypothesis.md", "People judge a sequence as more random the higher its alternation rate.\n")
    _write(cand / "CANDIDATE_BRIEF.md", "# Candidate Brief\n\nRefine one existing hypothesis.\n")
    _write(cand / "candidate.py", "def choice_prob():\n    return 0.55\n")
    _write(cand / "model_posterior.json", json.dumps({
        "posteriors": {"bayesian_fair_coin": 0.4, "iter0_candidate0": 0.6},
        "elpd_loo": {"bayesian_fair_coin": -68.9, "iter0_candidate0": -61.4}, "n_trials": 4}))
    _write(cand / "agent.jsonl", ANSI_TRANSCRIPT)

    _write_critique(ml / "iter_0" / "critique", with_results=with_critique_results)


def _write_critique(crit: Path, *, with_results: bool) -> None:
    _write(crit / "CRITIQUE_CONTEXT.md", "# Critique context\n\nIncumbent model: bayesian_fair_coin\n")
    _write(crit / "test_stats" / "alt_rate_gap.py",
           "# name: alternation_rate_gap\n"
           "# description: Mean alternation proportion of sequence A across trials.\n"
           "def test_statistic(df):\n    return float(df['p_alts_a'].mean())\n")
    _write(crit / "test_stats" / "lonely_stat.py",
           "def test_statistic(df):\n    \"\"\"Standard deviation of chosen side.\"\"\"\n    return float(df['chose_left'].std())\n")
    if with_results:
        _write(crit / "ppc_results.json", json.dumps({
            "model": "bayesian_fair_coin", "n_test_statistics": 2,
            "n_replicates": 200, "significance_alpha": 0.05, "n_significant": 1,
            "n_significant_fdr": 1,
            "results": [
                {"name": "alternation_rate_gap", "description": "Mean alternation proportion of A.",
                 "t_observed": 0.55, "null_mean": 0.50, "null_std": 0.02, "z_score": 2.5,
                 "p_value": 0.012, "p_value_one_sided": 0.006, "p_value_fdr": 0.024,
                 "p_value_is_floor": False, "significant": True, "significant_fdr": True,
                 "error": None, "code": "def test_statistic(df): ..."},
                {"name": "results_only_stat", "description": "A result with no source file.",
                 "t_observed": 2.1, "null_mean": 2.0, "null_std": 0.3, "z_score": 0.5,
                 "p_value": 0.6, "p_value_one_sided": 0.3, "p_value_fdr": 0.6,
                 "p_value_is_floor": False, "significant": False, "significant_fdr": False,
                 "error": None, "code": "def test_statistic(df): ..."},
            ]}))


def build_demo_tree(root: Path) -> Path:
    """Create the full data tree of runs and return the data-root path."""
    # A multi-experiment run with run-level analysis figures.
    demo = root / "outer_loop" / "demo"
    _write_bytes(demo / "analysis" / "loop_trajectory.png", PNG_1X1)
    _full_experiment(demo / "experiment1", with_critique_results=True)
    smoke = demo / "smoke"
    _write(smoke / "design" / "stimuli.json", json.dumps([{"sequence_a": "H", "sequence_b": "T"}]))
    _write(smoke / "experiment" / "config.json", json.dumps({"experiment_id": "demo_smoke"}))
    _write(smoke / "data" / "responses.csv",
           "participant_id,trial_index,chose_left,chose_right\n0,1,1,0\n")

    # A deeply-nested full experiment run.
    _full_experiment(root / "recovery" / "holdout_runs" / "cond_a" / "experiment1",
                     with_critique_results=False)

    # A bare model-loop run.
    mx = root / "recovery" / "confusion_runs" / "model_x"
    loop = mx / "loop"
    _write(loop / "model_posterior.json", json.dumps({
        "posteriors": {"prototype_similarity": 0.8, "bayesian_diagnosticity": 0.2},
        "elpd_loo": {"prototype_similarity": -50.0, "bayesian_diagnosticity": -55.0}, "n_trials": 20}))
    _write(loop / "models" / "prototype_similarity.py", "def choice_prob():\n    return 0.6\n")
    _write(loop / "report.md", "# loop report\n\nBest model: prototype_similarity\n")
    _write(mx / "data" / "responses.csv", "participant_id,chose_left\n0,1\n")
    # cache dir that must be ignored
    _write_bytes(mx / "mcmc_cache" / "fit.nc", b"\x00\x01")

    # A standalone run: theory + experiment-level critique, no model loop.
    ta = root / "thinkaloud" / "experiment1"
    _write(ta / "cognitive_models" / "models_manifest.yaml",
           "models:\n- name: dfs_sum_heuristic\n  rationale: Depth-first search summing.\n")
    _write(ta / "cognitive_models" / "dfs_sum_heuristic.py", "def choice_prob():\n    return 0.5\n")
    _write(ta / "critique" / "CRITIQUE_CONTEXT.md", "# Critique context\n\nIncumbent: dfs_sum_heuristic\n")
    _write(ta / "critique" / "test_stats" / "mean_solve_rate.py",
           "def test_statistic(df):\n    \"\"\"Mean solve rate across all problems.\"\"\"\n    return float(df['solved'].mean())\n")
    return root
