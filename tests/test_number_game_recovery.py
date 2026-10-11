"""A number-game holdout recovery cell runs end to end.

One seed is the ground truth: it simulates every participant's ratings at
fixed parameters, and the agentic loop starts from the other seed. Everything
runs for real at toy size (the EIG design, the simulator, the inner loop's
admission gates, MCMC, ELPD-LOO, the recovery scoring), except the coding
agents: the candidate agent is scripted to write the ground truth's own model,
which the loop should admit and the leakage audit should flag.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.runtime import coding_agent
from src.runtime.config import REPO_ROOT
from src.subjective_randomness.holdout_recovery import run_holdout_recovery_from_config

SEEDS_DIR = REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects" / "number_game" / "seed_models"
GT = "bigelow_piantadosi_independent"
START = "tenenbaum_rules_similarity"
GT_PARAMS = {
    GT: {"p_mix": 0.47, "rule_weights": list(np.full(42, 1 / 42))},
    START: {"lambda_rules": 0.5, "sigma": 10.0},
}


def _scripted_agent(calls):
    """The candidate agent writes the ground truth's model; the critic writes nothing."""

    def run(prompt, *, allowed_dirs, usage_label, **kwargs):
        calls.append(usage_label)
        if usage_label == "inner:candidate":
            out = Path(allowed_dirs[0])
            shutil.copyfile(SEEDS_DIR / f"{GT}.py", out / "candidate.py")
            (out / "hypothesis.md").write_text("Rules weighted one by one, and intervals.\n")
            (out / "model_name.txt").write_text("weighted_rules\n")
        return True, None

    return run


@pytest.mark.slow
def test_number_game_recovery_cell_end_to_end(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(coding_agent, "run_coding_agent", _scripted_agent(calls))
    config = {
        "project_id": "number_game",
        "seed_models_dir": str(SEEDS_DIR),
        "gt_models": GT_PARAMS,
        "n_experiments": 2,
        "n_participants": 2,
        "seed": 3,
        "design": {"n_eig": 6},
        "inner_loop": {"max_iterations": 1, "candidate_count": 1, "n_critique_proposals": 1},
        "agent": {"timeout_sec": 60, "backend": None, "model": "scripted"},
        "eval_pool": {"min_remaining": 100, "predict_max_draws": 20},
        "fit": {"draws": 200, "tune": 200, "chains": 2},
    }

    result = run_holdout_recovery_from_config(
        config,
        tmp_path / "config.yaml",
        tmp_path / "runs",
        gt_model_override=GT,
        cache_dir=tmp_path / "cache",
        summary_root=tmp_path / "summaries",
    )

    run_root = tmp_path / "runs" / "cell_1"
    [gt_run] = result["gt_runs"]

    # The loop starts from the other seed only.
    assert gt_run["experiments"][0]["manifest_models"][:1] == [START]
    assert GT not in (run_root / "experiment1" / "cognitive_models" / "models_manifest.yaml").read_text()

    # The agents see only the raw number-game columns, never the generator.
    with (run_root / "experiment1" / "data" / "responses.csv").open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == ["number_set", "target", "participant_id", "trial_index", "rating"]
    assert len(rows) == 6 * 2 and {r["rating"] for r in rows} <= {"0", "1"}

    # The scripted candidate (the ground truth's own code) is admitted...
    ledger = [
        json.loads(line)
        for line in (run_root / "experiment1" / "model_loop" / "attempted_hypotheses.jsonl").read_text().splitlines()
    ]
    assert any(e["outcome"] == "admitted" and e["name"] == "weighted_rules" for e in ledger)
    # ...and the leakage audit sees that it is a copy of the ground truth.
    assert gt_run["leakage"]["any_identical"] is True

    # Recovery is scored on every (set, target) cell the loop never trained on.
    n_cells = 255 * 100
    assert gt_run["n_eval_stimuli"] + gt_run["n_eval_dropped"] == n_cells
    assert gt_run["n_eval_dropped"] <= 2 * 6
    trajectory = gt_run["trajectory"]
    assert [row["experiment"] for row in trajectory][:1] == [1] and trajectory[-1]["experiment"] == 2
    for row in trajectory:
        assert np.isfinite(row["rmse"]) and np.isfinite(row["kl_regret"]) and -1 <= row["pearson_r"] <= 1
    assert set(gt_run["fitted_baseline"]["per_model"]) == {START}
