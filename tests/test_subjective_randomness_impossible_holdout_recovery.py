"""Tests for impossible-theory holdout recovery through the agentic loop.

The ground truth here is a deliberately weird generator (e.g. "more heads =>
more random-looking") that lives OUTSIDE the project seed pool. The integration
test drives ``run_impossible_holdout_recovery_from_config`` end to end with the
expensive seams stubbed out, asserting the observable contract: the agentic loop
is seeded with the *normal* project seed models, every ground-truth touchpoint
reads from the separate impossible-models directory, and the leakage audit is
robust to the impossible model having no pure-Python ``model_families``
counterpart.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

import src.subjective_randomness.holdout_recovery as holdout_recovery
from src.subjective_randomness.holdout_recovery import (
    _distinctive_param_names,
    build_eval_stimuli,
    leakage_check,
    run_impossible_holdout_recovery_from_config,
)
from src.subjective_randomness.model_recovery import p_left_fixed_params

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_MODELS_DIR = (
    REPO_ROOT / "src/pipelines/outer_loop/projects/subjective_randomness/seed_models"
)
IMPOSSIBLE_MODELS_DIR = REPO_ROOT / "src/subjective_randomness/impossible_models"

DESIGN_STIMULI = [
    {"sequence_a": "HTHTHT", "sequence_b": "HHHHHH", "eig": 0.9},
    {"sequence_a": "HHHTTT", "sequence_b": "HTHTHT", "eig": 0.5},
]


def _stub_spawn_cc_agent(calls):
    """Stand-in for the theory/design coding agents (mirrors the seed-model test)."""

    def spawn(agent_key, exp_dir, allowed_dirs=None, timeout_secs=900, backend=None):
        calls.append((agent_key, Path(exp_dir).name))
        if agent_key == "1_theory":
            exp_num = int(exp_dir.name.removeprefix("experiment"))
            prev_models = exp_dir.parent / f"experiment{exp_num - 1}" / "cognitive_models"
            dest = exp_dir / "cognitive_models"
            dest.mkdir(parents=True, exist_ok=True)
            for path in prev_models.iterdir():
                if path.is_file():
                    shutil.copyfile(path, dest / path.name)
        elif agent_key == "2_design":
            design_dir = exp_dir / "design"
            design_dir.mkdir(parents=True, exist_ok=True)
            (design_dir / "stimuli.json").write_text(
                json.dumps(DESIGN_STIMULI), encoding="utf-8"
            )
        else:
            raise AssertionError(f"Unexpected agent spawned: {agent_key}")
        return True, "ok"

    return spawn


def _stub_generate_responses(calls):
    """Records ``models_dir`` so the test can assert the GT dir is threaded in."""

    def generate(model_name, models_dir, stimuli, params, n_participants, *, seed=0,
                 generator="pymc"):
        calls.append(
            {"model_name": model_name, "models_dir": Path(models_dir), "seed": seed}
        )
        rows = []
        for participant in range(n_participants):
            for trial_index, stim in enumerate(stimuli):
                rows.append(
                    {
                        "participant_id": participant,
                        "trial_index": trial_index,
                        "sequence_a": stim["sequence_a"],
                        "sequence_b": stim["sequence_b"],
                        "chose_left": (participant + trial_index) % 2,
                        "generating_model": model_name,
                    }
                )
        return rows

    return generate


def _stub_inner_loop(history_best="encoding_compressibility"):
    def run(exp_dir, *, max_iterations, candidate_count, fit_kwargs=None,
            backend=None, cache_dir=None, project_id=None, agent_timeout_sec=900):
        loop_dir = exp_dir / "model_loop"
        models_dir = loop_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        cognitive_dir = exp_dir / "cognitive_models"
        for path in cognitive_dir.glob("*.py"):
            shutil.copyfile(path, models_dir / path.name)

        posteriors = {history_best: 0.8, "bayesian_diagnosticity": 0.2}
        elpd = {history_best: -10.0, "bayesian_diagnosticity": -12.0}
        history = [
            {"step": 0, "iteration": None, "best_model": history_best,
             "posteriors": posteriors, "elpd_loo": elpd},
            {"step": 1, "iteration": 0, "best_model": history_best,
             "posteriors": posteriors, "elpd_loo": elpd},
        ]
        (loop_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
        (loop_dir / "model_posterior.json").write_text(
            json.dumps({"posteriors": posteriors, "elpd_loo": elpd, "n_trials": 4}),
            encoding="utf-8",
        )
        (loop_dir / "report.md").write_text("# stub report\n", encoding="utf-8")
        (loop_dir / "responses.csv").write_text("chose_left\n1\n", encoding="utf-8")

        # Mirror _export_inner_loop_model: best model into cognitive_models + manifest.
        shutil.copyfile(
            cognitive_dir / f"{history_best}.py",
            cognitive_dir / "inner_loop_model.py",
        )
        manifest_path = cognitive_dir / "models_manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        manifest["models"] = [
            m for m in manifest["models"] if m.get("name") != "inner_loop_model"
        ] + [{"name": "inner_loop_model", "rationale": "stub"}]
        manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        return loop_dir

    return run


class _FakeFitted:
    model = None

    def predict_p_left(self, stim_data):
        return np.linspace(0.1, 0.9, stim_data["n"])


def test_impossible_holdout_recovery_from_config_end_to_end_with_stub_agents(
    tmp_path, monkeypatch
):
    agent_calls = []
    collect_calls = []

    monkeypatch.setattr(
        holdout_recovery, "spawn_cc_agent", _stub_spawn_cc_agent(agent_calls)
    )
    monkeypatch.setattr(
        holdout_recovery, "generate_responses", _stub_generate_responses(collect_calls)
    )
    monkeypatch.setattr(
        holdout_recovery, "run_inner_model_loop_programmatic", _stub_inner_loop()
    )
    # The GT reference p_left is stubbed (varied, so correlation is defined);
    # the impossible PyMC model file is exercised by the unit tests, not here.
    monkeypatch.setattr(
        holdout_recovery,
        "p_left_fixed_params",
        lambda model_name, models_dir, stimuli, params, **kw: np.linspace(
            0.1, 0.9, len(stimuli)
        ),
    )
    monkeypatch.setattr(
        holdout_recovery, "make_stim_data", lambda model, rows: {"n": len(rows)}
    )
    monkeypatch.setattr(holdout_recovery, "pm_data_inputs", lambda model: [])
    monkeypatch.setattr(
        holdout_recovery,
        "fit_model",
        lambda name, models_dir, responses_path, *, cache_dir=None, **kw: _FakeFitted(),
    )

    config = {
        "project_id": "subjective_randomness",
        "seed_models_dir": str(SEED_MODELS_DIR),
        "gt_models_dir": str(IMPOSSIBLE_MODELS_DIR),
        "gt_models": {"more_heads_more_random": {"beta": 4.0, "side_bias": 0.0}},
        "n_experiments": 2,
        "n_participants": 3,
        "seed": 5,
        "inner_loop": {"max_iterations": 1, "candidate_count": 1},
        "agent": {"timeout_sec": 60, "backend": None},
        "eval_pool": {"n_pairs": 40, "lengths": [6], "seed": 11, "min_remaining": 5},
        "fit": {"draws": 10, "tune": 10, "chains": 1},
    }

    result = run_impossible_holdout_recovery_from_config(
        config,
        tmp_path / "config.yaml",
        tmp_path / "runs",
        cache_dir=tmp_path / "cache",
    )

    # The agentic loop is seeded with the NORMAL project seed models; the
    # impossible ground truth never enters experiment 1's seed set.
    run_root = tmp_path / "runs" / "more_heads_more_random"
    exp1_models = run_root / "experiment1" / "cognitive_models"
    assert not (exp1_models / "more_heads_more_random.py").exists()
    seeded = yaml.safe_load(
        (exp1_models / "models_manifest.yaml").read_text(encoding="utf-8")
    )
    seeded_names = {m["name"] for m in seeded["models"]}
    assert "more_heads_more_random" not in seeded_names
    assert {
        "prototype_similarity",
        "bayesian_diagnosticity",
        "encoding_compressibility",
    } <= seeded_names

    # Every response is generated from the impossible GT, read from the SEPARATE
    # impossible-models directory (not the seed dir).
    assert [c["model_name"] for c in collect_calls] == ["more_heads_more_random"] * 2
    assert all(c["models_dir"] == IMPOSSIBLE_MODELS_DIR for c in collect_calls)

    # One trajectory row per inner-loop history step per experiment.
    assert len(result["gt_runs"]) == 1
    gt_run = result["gt_runs"][0]
    assert gt_run["gt_model"] == "more_heads_more_random"
    assert [row["global_step"] for row in gt_run["trajectory"]] == [0, 1, 2, 3]

    # The fitted-seed baseline fits all three normal seeds (none excluded, since
    # the impossible GT is not in the project seed set).
    assert set(gt_run["fitted_baseline"]["per_model"]) == {
        "prototype_similarity",
        "bayesian_diagnosticity",
        "encoding_compressibility",
    }

    # Leakage is audited and robust to the impossible GT having no model_families
    # counterpart: no distinctive params to mention, no identical copy.
    assert gt_run["leakage"]["any_identical"] is False
    assert gt_run["leakage"]["any_mention"] is False
    assert (run_root / "trajectory.json").exists()
    assert (run_root / "eval_stimuli.json").exists()


# ── impossible-theory PyMC models ───────────────────────────────────

IMPOSSIBLE_PARAMS = {"beta": 4.0, "side_bias": 0.0}


@pytest.mark.parametrize(
    "model_name, more, less",
    [
        ("more_heads_more_random", "HHHHHT", "HTHTHT"),  # 5 vs 3 heads
        ("fewer_heads_more_random", "HTHTHT", "HHHHHT"),  # 3 vs 5 heads
        ("longer_runs_more_random", "HHHHHH", "HTHTHT"),  # one long run vs none
        ("more_imbalance_more_random", "HHHHHH", "HHHTTT"),  # all-H vs balanced
    ],
)
def test_impossible_model_prefers_its_feature(model_name, more, less):
    # The sequence with more of the impossible theory's feature is judged "more
    # random" (p_left > 0.5); the reverse pairing is judged less random.
    forward = p_left_fixed_params(
        model_name,
        IMPOSSIBLE_MODELS_DIR,
        [{"sequence_a": more, "sequence_b": less}],
        IMPOSSIBLE_PARAMS,
    )
    backward = p_left_fixed_params(
        model_name,
        IMPOSSIBLE_MODELS_DIR,
        [{"sequence_a": less, "sequence_b": more}],
        IMPOSSIBLE_PARAMS,
    )
    assert forward[0] > 0.5
    assert backward[0] < 0.5


def test_impossible_model_p_left_varies_across_stimuli():
    # A varied stimulus list yields non-constant p_left, so the downstream
    # Pearson correlation against this ground truth is well defined.
    stimuli = [
        {"sequence_a": "HHHHHH", "sequence_b": "HTHTHT"},
        {"sequence_a": "HHHTTT", "sequence_b": "HHTHTT"},
        {"sequence_a": "HTHTHT", "sequence_b": "HHHHHH"},
    ]
    p_left = p_left_fixed_params(
        "more_heads_more_random", IMPOSSIBLE_MODELS_DIR, stimuli, IMPOSSIBLE_PARAMS
    )
    assert len(set(np.round(p_left, 6))) > 1


def test_impossible_model_requires_exact_params():
    stimuli = [{"sequence_a": "HHHHHT", "sequence_b": "HTHTHT"}]
    # Exactly {beta, side_bias} is accepted.
    p_left_fixed_params(
        "more_heads_more_random", IMPOSSIBLE_MODELS_DIR, stimuli, IMPOSSIBLE_PARAMS
    )
    # An unexpected extra parameter fails loudly.
    with pytest.raises(ValueError, match="free parameters"):
        p_left_fixed_params(
            "more_heads_more_random",
            IMPOSSIBLE_MODELS_DIR,
            stimuli,
            {"beta": 4.0, "side_bias": 0.0, "theta": 1.0},
        )


# ── leakage audit / param resolution robustness ─────────────────────


def test_distinctive_param_names_empty_for_impossible_model():
    # An impossible model has no model_families counterpart, so there are no
    # distinctive params to leak (and no crash).
    assert _distinctive_param_names("more_heads_more_random") == set()
    # An existing family still reports its distinctive params (no regression).
    assert _distinctive_param_names("prototype_similarity")


def test_leakage_check_robust_to_missing_model_family(tmp_path):
    run_root = tmp_path / "run"
    models_dir = run_root / "experiment1" / "cognitive_models"
    models_dir.mkdir(parents=True)
    (models_dir / "candidate.py").write_text(
        "# beta and side_bias only\n", encoding="utf-8"
    )
    result = leakage_check(
        run_root,
        "more_heads_more_random",
        seed_models_dir=SEED_MODELS_DIR,
        n_experiments=1,
        gt_models_dir=IMPOSSIBLE_MODELS_DIR,
    )
    assert result["any_identical"] is False
    assert result["any_mention"] is False
    assert result["any_gt_named"] is False


# ── wrapper guards ──────────────────────────────────────────────────


def test_impossible_wrapper_requires_explicit_params(tmp_path):
    config = {
        "project_id": "subjective_randomness",
        "seed_models_dir": str(SEED_MODELS_DIR),
        "gt_models_dir": str(IMPOSSIBLE_MODELS_DIR),
        "gt_models": {"more_heads_more_random": None},
    }
    with pytest.raises(ValueError, match="explicit params"):
        run_impossible_holdout_recovery_from_config(
            config, tmp_path / "config.yaml", tmp_path / "runs"
        )


def test_impossible_wrapper_requires_gt_models_dir(tmp_path):
    config = {
        "seed_models_dir": str(SEED_MODELS_DIR),
        "gt_models": {"more_heads_more_random": {"beta": 4.0, "side_bias": 0.0}},
    }
    with pytest.raises(KeyError, match="gt_models_dir"):
        run_impossible_holdout_recovery_from_config(
            config, tmp_path / "config.yaml", tmp_path / "runs"
        )


# ── exhaustive eval pool + posterior thinning ───────────────────────


def test_build_eval_stimuli_exhaustive_enumerates_all_pairs(tmp_path):
    run_root = tmp_path / "run"
    data_dir = run_root / "experiment1" / "data"
    data_dir.mkdir(parents=True)
    # One length-6 training pair — it cannot overlap the length-2/3 eval space.
    (data_dir / "responses.csv").write_text(
        "sequence_a,sequence_b,chose_left\nHTHTHT,HHHHHH,1\n", encoding="utf-8"
    )
    info = build_eval_stimuli(
        run_root,
        n_experiments=1,
        n_pairs=0,
        lengths=[2, 3],
        seed=0,
        min_remaining=1,
        exhaustive=True,
    )
    # Every pair over the 4 + 8 = 12 sequences, cross-length included:
    # C(12, 2) = 66, none dropped (the training pair is length-6).
    assert len(info["stimuli"]) == 66
    assert info["n_dropped"] == 0
    assert all(len(s["sequence_a"]) in (2, 3) for s in info["stimuli"])
    assert any(
        len(s["sequence_a"]) != len(s["sequence_b"]) for s in info["stimuli"]
    )


def test_impossible_holdout_exhaustive_eval_thins_posterior(tmp_path, monkeypatch):
    predict_max_draws_seen = []

    class _RecordingFitted:
        model = None

        def predict_p_left(self, stim_data, *, max_draws=None):
            predict_max_draws_seen.append(max_draws)
            return np.linspace(0.1, 0.9, stim_data["n"])

    monkeypatch.setattr(holdout_recovery, "spawn_cc_agent", _stub_spawn_cc_agent([]))
    monkeypatch.setattr(
        holdout_recovery, "generate_responses", _stub_generate_responses([])
    )
    monkeypatch.setattr(
        holdout_recovery, "run_inner_model_loop_programmatic", _stub_inner_loop()
    )
    monkeypatch.setattr(
        holdout_recovery,
        "p_left_fixed_params",
        lambda model_name, models_dir, stimuli, params, **kw: np.linspace(
            0.1, 0.9, len(stimuli)
        ),
    )
    monkeypatch.setattr(
        holdout_recovery, "make_stim_data", lambda model, rows: {"n": len(rows)}
    )
    monkeypatch.setattr(holdout_recovery, "pm_data_inputs", lambda model: [])
    monkeypatch.setattr(
        holdout_recovery,
        "fit_model",
        lambda name, models_dir, responses_path, *, cache_dir=None, **kw: (
            _RecordingFitted()
        ),
    )

    config = {
        "project_id": "subjective_randomness",
        "seed_models_dir": str(SEED_MODELS_DIR),
        "gt_models_dir": str(IMPOSSIBLE_MODELS_DIR),
        "gt_models": {"more_heads_more_random": {"beta": 4.0, "side_bias": 0.0}},
        "n_experiments": 2,
        "n_participants": 3,
        "seed": 5,
        "inner_loop": {"max_iterations": 1, "candidate_count": 1},
        "agent": {"timeout_sec": 60, "backend": None},
        "eval_pool": {
            "exhaustive": True,
            "lengths": [2, 3],
            "min_remaining": 1,
            "predict_max_draws": 7,
        },
        "fit": {"draws": 10, "tune": 10, "chains": 1},
    }

    result = run_impossible_holdout_recovery_from_config(
        config,
        tmp_path / "config.yaml",
        tmp_path / "runs",
        cache_dir=tmp_path / "cache",
    )

    gt_run = result["gt_runs"][0]
    # Exhaustive enumeration over lengths 2,3 pools 12 sequences into C(12,2)=66
    # pairs (cross-length included); the length-6 training pairs cannot overlap,
    # so none are dropped.
    assert gt_run["n_eval_stimuli"] == 66
    assert gt_run["n_eval_dropped"] == 0
    # Every held-out prediction thinned the posterior to the configured budget.
    assert predict_max_draws_seen
    assert all(m == 7 for m in predict_max_draws_seen)
    # The eval settings are recorded for provenance.
    assert result["eval_pool"]["exhaustive"] is True
    assert result["eval_pool"]["predict_max_draws"] == 7
