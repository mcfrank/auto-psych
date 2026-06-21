"""Tests for ground-truth holdout recovery through the agentic loop.

The integration test drives `run_holdout_recovery_from_config` end to end with
the expensive seams stubbed out (coding agents, MCMC, fixed-param prior
predictive), asserting the observable contract: the held-out model is excluded
from experiment 1's seed set, agents run in pipeline order, and the trajectory
correlates every inner-loop history step against the ground truth on a
held-out stimulus set.
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
    TRAJECTORY_COLUMNS,
    build_eval_stimuli,
    collect_trained_pairs,
    evaluate_trajectory,
    fitted_seed_baseline_correlation,
    leakage_check,
    reevaluate_trajectories,
    run_holdout_experiments,
    run_holdout_recovery_from_config,
    seed_baseline_correlation,
    trajectory_tidy_rows,
)
from src.subjective_randomness.recover import pearson_r
from src.subjective_randomness.stimulus_design import generate_candidate_pool

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_MODELS_DIR = (
    REPO_ROOT / "src/pipelines/outer_loop/projects/subjective_randomness/seed_models"
)

DESIGN_STIMULI = [
    {"sequence_a": "HTHTHT", "sequence_b": "HHHHHH", "eig": 0.9},
    {"sequence_a": "HHHTTT", "sequence_b": "HTHTHT", "eig": 0.5},
]


def _stub_spawn_cc_agent(calls):
    """Stand-in for the theory/design coding agents.

    The theory stub (experiments >= 2) carries the previous experiment's model
    set forward; the design stub writes a valid EIG-annotated stimuli.json.
    """

    def spawn(agent_key, exp_dir, allowed_dirs=None, timeout_secs=900, backend=None, prompt_key=None, repair_feedback=None):
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
    def generate(model_name, models_dir, stimuli, params, n_participants, *, seed=0,
                 generator="pymc"):
        calls.append({"model_name": model_name, "seed": seed, "params": dict(params)})
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


def test_holdout_recovery_from_config_end_to_end_with_stub_agents(tmp_path, monkeypatch):
    agent_calls = []
    collect_calls = []
    fit_calls = []

    monkeypatch.setattr(
        holdout_recovery, "spawn_cc_agent", _stub_spawn_cc_agent(agent_calls)
    )
    monkeypatch.setattr(
        holdout_recovery, "generate_responses", _stub_generate_responses(collect_calls)
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
    # No model in these stubs indexes a participant random effect.
    monkeypatch.setattr(holdout_recovery, "pm_data_inputs", lambda model: [])

    def fake_fit_model(name, models_dir, responses_path, *, cache_dir=None, **kw):
        fit_calls.append({"name": name, "cache_dir": cache_dir})
        return _FakeFitted()

    monkeypatch.setattr(holdout_recovery, "fit_model", fake_fit_model)

    config = {
        "project_id": "subjective_randomness",
        "seed_models_dir": str(SEED_MODELS_DIR),
        "gt_models": ["prototype_similarity"],
        "n_experiments": 2,
        "n_participants": 3,
        "seed": 5,
        "inner_loop": {"max_iterations": 1, "candidate_count": 1},
        "agent": {"timeout_sec": 60, "backend": None},
        "eval_pool": {"n_pairs": 40, "lengths": [6], "seed": 11, "min_remaining": 5},
        "fit": {"draws": 10, "tune": 10, "chains": 1},
    }

    result = run_holdout_recovery_from_config(
        config,
        tmp_path / "config.yaml",
        tmp_path / "runs",
        cache_dir=tmp_path / "cache",
    )

    # The held-out model never enters experiment 1's seed set.
    run_root = tmp_path / "runs" / "prototype_similarity"
    exp1_models = run_root / "experiment1" / "cognitive_models"
    assert not (exp1_models / "prototype_similarity.py").exists()
    seeded = yaml.safe_load(
        (exp1_models / "models_manifest.yaml").read_text(encoding="utf-8")
    )
    seeded_names = {m["name"] for m in seeded["models"]}
    assert "prototype_similarity" not in seeded_names
    assert {"bayesian_diagnosticity", "encoding_compressibility"} <= seeded_names

    # Agents run in pipeline order: design only in exp 1 (seeded theory), then
    # theory + design in exp 2.
    assert agent_calls == [
        ("2_design", "experiment1"),
        ("1_theory", "experiment2"),
        ("2_design", "experiment2"),
    ]

    # Collection always samples from the held-out ground truth, with a fresh
    # per-experiment seed offset.
    assert [c["model_name"] for c in collect_calls] == ["prototype_similarity"] * 2
    assert [c["seed"] for c in collect_calls] == [6, 7]

    # Trajectory: one row per history step per experiment, monotone global_step,
    # and (with identical stub predictions) perfect correlation.
    assert len(result["gt_runs"]) == 1
    gt_run = result["gt_runs"][0]
    assert gt_run["gt_model"] == "prototype_similarity"
    trajectory = gt_run["trajectory"]
    assert [row["global_step"] for row in trajectory] == [0, 1, 2, 3]
    assert [row["experiment"] for row in trajectory] == [1, 1, 2, 2]
    assert [row["iteration"] for row in trajectory] == [None, 0, None, 0]
    assert all(row["best_model"] == "encoding_compressibility" for row in trajectory)
    assert all(row["pearson_r"] == pytest.approx(1.0) for row in trajectory)
    assert all(row["rmse"] == pytest.approx(0.0) for row in trajectory)
    # The Bayesian model average (identical stub predictions) also recovers it.
    assert all(row["pearson_r_bma"] == pytest.approx(1.0) for row in trajectory)
    assert all(row["rmse_bma"] == pytest.approx(0.0) for row in trajectory)
    # The fitted-seed baseline (one flat number, seeds fit on all data) recovers
    # it too, since every stub prediction is identical.
    assert set(gt_run["fitted_baseline"]["per_model"]) == {
        "bayesian_diagnosticity",
        "encoding_compressibility",
        "window_typicality",
    }
    assert gt_run["fitted_baseline"]["mean_r"] == pytest.approx(1.0)

    # Evaluation refits go through the shared MCMC cache. The BMA fits every
    # posterior-weighted model, not just the single best one.
    assert all(c["cache_dir"] == tmp_path / "cache" for c in fit_calls)
    assert {c["name"] for c in fit_calls} == {
        "encoding_compressibility",
        "bayesian_diagnosticity",
        "window_typicality",
    }

    # The no-learning baseline averages the other seed models (default params)
    # against the GT; with identical stub predictions every correlation is 1.
    assert set(gt_run["baseline"]["per_model"]) == {
        "bayesian_diagnosticity",
        "encoding_compressibility",
        "window_typicality",
    }
    assert gt_run["baseline"]["mean_r"] == pytest.approx(1.0)

    # The eval set is recorded, sized, and leakage-audited.
    assert gt_run["n_eval_stimuli"] + gt_run["n_eval_dropped"] == 40
    assert gt_run["n_eval_stimuli"] > 0
    assert gt_run["leakage"]["any_identical"] is False
    assert (run_root / "trajectory.json").exists()
    assert (run_root / "eval_stimuli.json").exists()

    # Per-experiment model sets are recorded for transparency.
    assert len(gt_run["experiments"]) == 2
    assert "inner_loop_model" in gt_run["experiments"][1]["manifest_models"]


# ── harness error path ──────────────────────────────────────────────


def test_run_holdout_experiments_raises_on_invalid_agent_output(tmp_path, monkeypatch):
    # A design agent that produces nothing must stop the run loudly, naming the
    # stage, instead of collecting data against a missing design.
    monkeypatch.setattr(
        holdout_recovery, "spawn_cc_agent", lambda *a, **kw: (True, "did nothing")
    )

    with pytest.raises(RuntimeError, match="2_design"):
        run_holdout_experiments(
            "prototype_similarity",
            {"theta_alt": 0.65, "alt_weight": 0.55, "beta": 4.0, "side_bias": 0.0},
            tmp_path / "run",
            seed_models_dir=SEED_MODELS_DIR,
            n_experiments=1,
            n_participants=2,
            inner_loop_iterations=0,
            candidate_count=0,
            fit_kwargs={},
            seed=0,
        )


# ── resume ──────────────────────────────────────────────────────────


def _complete_experiment_on_disk(run_root, exp_num, *, with_model_loop=True):
    """Materialize an experiment dir that passes every stage validator."""
    exp_dir = run_root / f"experiment{exp_num}"
    models_dir = exp_dir / "cognitive_models"
    models_dir.mkdir(parents=True)
    for name in ("bayesian_diagnosticity", "encoding_compressibility"):
        shutil.copyfile(SEED_MODELS_DIR / f"{name}.py", models_dir / f"{name}.py")
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "name": "bayesian_diagnosticity",
                        "rationale": "Fair-coin diagnosticity hypothesis.",
                    },
                    {
                        "name": "encoding_compressibility",
                        "rationale": "Compressibility-penalty hypothesis.",
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    design_dir = exp_dir / "design"
    design_dir.mkdir()
    (design_dir / "stimuli.json").write_text(
        json.dumps(DESIGN_STIMULI), encoding="utf-8"
    )
    data_dir = exp_dir / "data"
    data_dir.mkdir()
    (data_dir / "responses.csv").write_text(
        "participant_id,trial_index,sequence_a,sequence_b,chose_left\n"
        "0,0,HTHTHT,HHHHHH,1\n",
        encoding="utf-8",
    )
    if with_model_loop:
        loop_dir = exp_dir / "model_loop"
        (loop_dir / "models").mkdir(parents=True)
        (loop_dir / "model_posterior.json").write_text(
            json.dumps({"posteriors": {"encoding_compressibility": 1.0},
                        "elpd_loo": {"encoding_compressibility": -1.0},
                        "n_trials": 1}),
            encoding="utf-8",
        )
        (loop_dir / "report.md").write_text("# done\n", encoding="utf-8")
        (loop_dir / "responses.csv").write_text("chose_left\n1\n", encoding="utf-8")
        (loop_dir / "history.json").write_text(
            json.dumps([_history_step(0, None, "encoding_compressibility")]),
            encoding="utf-8",
        )
        shutil.copyfile(
            models_dir / "encoding_compressibility.py",
            models_dir / "inner_loop_model.py",
        )
    return exp_dir


def test_run_holdout_experiments_refuses_existing_dir_without_resume(tmp_path):
    run_root = tmp_path / "run"
    (run_root / "experiment1").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="resume"):
        run_holdout_experiments(
            "prototype_similarity",
            {"theta_alt": 0.65, "alt_weight": 0.55, "beta": 4.0, "side_bias": 0.0},
            run_root,
            seed_models_dir=SEED_MODELS_DIR,
            n_experiments=1,
            n_participants=2,
            inner_loop_iterations=0,
            candidate_count=0,
            fit_kwargs={},
            seed=0,
        )


def test_run_holdout_experiments_resume_skips_valid_stages_and_reruns_invalid(
    tmp_path, monkeypatch
):
    # experiment1 is complete; experiment2 stopped after the theory stage (the
    # design agent failed). Resume must rerun only experiment2's design,
    # collect, and inner loop — never respawn work whose output validates.
    run_root = tmp_path / "run"
    _complete_experiment_on_disk(run_root, 1)
    exp2_models = run_root / "experiment2" / "cognitive_models"
    exp2_models.mkdir(parents=True)
    for name in ("bayesian_diagnosticity", "encoding_compressibility"):
        shutil.copyfile(SEED_MODELS_DIR / f"{name}.py", exp2_models / f"{name}.py")
    (exp2_models / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [
                {"name": "bayesian_diagnosticity",
                 "rationale": "Fair-coin diagnosticity hypothesis."},
                {"name": "encoding_compressibility",
                 "rationale": "Compressibility-penalty hypothesis."},
            ]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    agent_calls = []
    collect_calls = []
    loop_calls = []

    monkeypatch.setattr(
        holdout_recovery, "spawn_cc_agent", _stub_spawn_cc_agent(agent_calls)
    )
    monkeypatch.setattr(
        holdout_recovery, "generate_responses", _stub_generate_responses(collect_calls)
    )
    inner_stub = _stub_inner_loop()

    def counting_inner_loop(exp_dir, **kwargs):
        loop_calls.append(exp_dir.name)
        return inner_stub(exp_dir, **kwargs)

    monkeypatch.setattr(
        holdout_recovery, "run_inner_model_loop_programmatic", counting_inner_loop
    )

    run_holdout_experiments(
        "prototype_similarity",
        {"theta_alt": 0.65, "alt_weight": 0.55, "beta": 4.0, "side_bias": 0.0},
        run_root,
        seed_models_dir=SEED_MODELS_DIR,
        n_experiments=2,
        n_participants=2,
        inner_loop_iterations=0,
        candidate_count=0,
        fit_kwargs={},
        seed=5,
        resume=True,
    )

    assert agent_calls == [("2_design", "experiment2")]
    assert [c["seed"] for c in collect_calls] == [7]  # seed + exp_num, exp2 only
    assert loop_calls == ["experiment2"]


def test_run_holdout_experiments_resume_wipes_partial_model_loop(
    tmp_path, monkeypatch
):
    # A crashed inner loop leaves a partial model_loop whose manifest would be
    # reset on rerun, orphaning stale candidates; resume must wipe it (it is
    # fully regenerable — MCMC fits live in the shared cache) and rerun fresh.
    run_root = tmp_path / "run"
    exp_dir = _complete_experiment_on_disk(run_root, 1, with_model_loop=False)
    stale = exp_dir / "model_loop" / "models" / "iter0_candidate0.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("# stale partial candidate\n", encoding="utf-8")

    inner_stub = _stub_inner_loop()
    seen = {}

    def checking_inner_loop(exp_dir, **kwargs):
        seen["stale_present_at_call"] = stale.exists()
        return inner_stub(exp_dir, **kwargs)

    monkeypatch.setattr(
        holdout_recovery, "run_inner_model_loop_programmatic", checking_inner_loop
    )
    monkeypatch.setattr(holdout_recovery, "spawn_cc_agent", _stub_spawn_cc_agent([]))
    monkeypatch.setattr(
        holdout_recovery, "generate_responses", _stub_generate_responses([])
    )

    run_holdout_experiments(
        "prototype_similarity",
        {"theta_alt": 0.65, "alt_weight": 0.55, "beta": 4.0, "side_bias": 0.0},
        run_root,
        seed_models_dir=SEED_MODELS_DIR,
        n_experiments=1,
        n_participants=2,
        inner_loop_iterations=0,
        candidate_count=0,
        fit_kwargs={},
        seed=0,
        resume=True,
    )

    assert seen["stale_present_at_call"] is False


def test_from_config_resume_skips_completed_gt_runs(tmp_path, monkeypatch):
    completed = {
        "gt_model": "prototype_similarity",
        "params": {"theta_alt": 0.65},
        "run_root": "x",
        "n_eval_stimuli": 10,
        "n_eval_dropped": 0,
        "trajectory": [{"experiment": 1, "step": 0, "iteration": None,
                        "global_step": 0, "best_model": "a", "pearson_r": 0.9,
                        "rmse": 0.1}],
        "leakage": {"files": [], "any_identical": False, "any_mention": False,
                    "any_gt_named": False},
        "experiments": [{"experiment": 1, "manifest_models": ["a"]}],
    }
    run_root = tmp_path / "runs" / "prototype_similarity"
    run_root.mkdir(parents=True)
    (run_root / "trajectory.json").write_text(json.dumps(completed), encoding="utf-8")

    def tripwire(*args, **kwargs):
        raise AssertionError("completed GT run must not re-run any work")

    for seam in ("spawn_cc_agent", "seed_experiment_models_from_project",
                 "generate_responses", "run_inner_model_loop_programmatic",
                 "p_left_fixed_params", "fit_model"):
        monkeypatch.setattr(holdout_recovery, seam, tripwire)

    config = {
        "seed_models_dir": str(SEED_MODELS_DIR),
        "gt_models": ["prototype_similarity"],
        "n_experiments": 1,
    }
    result = run_holdout_recovery_from_config(
        config, tmp_path / "config.yaml", tmp_path / "runs", resume=True
    )
    assert result["gt_runs"] == [completed]


def test_from_config_resume_rejects_stale_trajectory_experiment_count(
    tmp_path, monkeypatch
):
    stale = {
        "gt_model": "prototype_similarity",
        "trajectory": [],
        "experiments": [{"experiment": 1, "manifest_models": []}],
    }
    run_root = tmp_path / "runs" / "prototype_similarity"
    run_root.mkdir(parents=True)
    (run_root / "trajectory.json").write_text(json.dumps(stale), encoding="utf-8")
    monkeypatch.setattr(
        holdout_recovery, "spawn_cc_agent",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    config = {
        "seed_models_dir": str(SEED_MODELS_DIR),
        "gt_models": ["prototype_similarity"],
        "n_experiments": 2,  # config asks for more experiments than recorded
    }
    with pytest.raises(ValueError, match="trajectory.json"):
        run_holdout_recovery_from_config(
            config, tmp_path / "config.yaml", tmp_path / "runs", resume=True
        )


# ── design EIG fallback (agent backgrounded the scoring) ────────────

PROTOTYPE_GT_PARAMS = {
    "theta_alt": 0.65,
    "alt_weight": 0.55,
    "beta": 4.0,
    "side_bias": 0.0,
}
CANDIDATE_POOL = [
    {"sequence_a": "HTHTHT", "sequence_b": "HHHHHH"},
    {"sequence_a": "HHHTTT", "sequence_b": "HTHTHT"},
    {"sequence_a": "HHTTHH", "sequence_b": "TTTTTT"},
]


def _stub_annotate_eig(calls):
    def annotate(candidates, models_dir, registry_path=None, *, featurize_path=None,
                 n_samples=200, seed=42):
        calls.append({"n": len(candidates), "models_dir": Path(models_dir),
                      "featurize_path": featurize_path})
        # Mirror the real annotate: add a descending eig and sort by it.
        scored = [
            {**c, "eig": round(1.0 - i * 0.1, 6)} for i, c in enumerate(candidates)
        ]
        scored.sort(key=lambda x: -x["eig"])
        return scored

    return annotate


def _spawn_writing_candidates(calls, pool=CANDIDATE_POOL):
    """Design agent that writes only candidates.json (backgrounded EIG, no stimuli)."""

    def spawn(agent_key, exp_dir, allowed_dirs=None, timeout_secs=900, backend=None, prompt_key=None, repair_feedback=None):
        calls.append((agent_key, Path(exp_dir).name))
        if agent_key == "2_design":
            design_dir = exp_dir / "design"
            design_dir.mkdir(parents=True, exist_ok=True)
            (design_dir / "candidates.json").write_text(
                json.dumps(pool), encoding="utf-8"
            )
        return True, "ok"

    return spawn


def test_design_stage_scores_candidates_when_agent_leaves_no_stimuli(
    tmp_path, monkeypatch
):
    run_root = tmp_path / "run"
    annotate_calls = []
    monkeypatch.setattr(
        holdout_recovery, "spawn_cc_agent", _spawn_writing_candidates([])
    )
    monkeypatch.setattr(
        holdout_recovery, "annotate_eig", _stub_annotate_eig(annotate_calls)
    )
    monkeypatch.setattr(
        holdout_recovery, "generate_responses", _stub_generate_responses([])
    )
    monkeypatch.setattr(
        holdout_recovery, "run_inner_model_loop_programmatic", _stub_inner_loop()
    )

    run_holdout_experiments(
        "prototype_similarity",
        PROTOTYPE_GT_PARAMS,
        run_root,
        seed_models_dir=SEED_MODELS_DIR,
        n_experiments=1,
        n_participants=2,
        inner_loop_iterations=0,
        candidate_count=0,
        fit_kwargs={},
        seed=0,
    )

    # The harness scored the full candidate pool against this experiment's models.
    assert len(annotate_calls) == 1
    assert annotate_calls[0]["n"] == len(CANDIDATE_POOL)
    assert annotate_calls[0]["models_dir"] == (
        run_root / "experiment1" / "cognitive_models"
    )
    stimuli = json.loads(
        (run_root / "experiment1" / "design" / "stimuli.json").read_text(
            encoding="utf-8"
        )
    )
    assert stimuli and all("eig" in s for s in stimuli)
    assert stimuli == sorted(stimuli, key=lambda s: -s["eig"])  # EIG-descending


def test_design_stage_uses_agent_stimuli_without_rescoring(tmp_path, monkeypatch):
    # When the agent does produce stimuli.json, the harness must not re-score.
    run_root = tmp_path / "run"
    monkeypatch.setattr(
        holdout_recovery, "spawn_cc_agent", _stub_spawn_cc_agent([])
    )

    def tripwire(*args, **kwargs):
        raise AssertionError("annotate must not run when the agent wrote stimuli.json")

    monkeypatch.setattr(holdout_recovery, "annotate_eig", tripwire)
    monkeypatch.setattr(
        holdout_recovery, "generate_responses", _stub_generate_responses([])
    )
    monkeypatch.setattr(
        holdout_recovery, "run_inner_model_loop_programmatic", _stub_inner_loop()
    )

    run_holdout_experiments(
        "prototype_similarity",
        PROTOTYPE_GT_PARAMS,
        run_root,
        seed_models_dir=SEED_MODELS_DIR,
        n_experiments=1,
        n_participants=2,
        inner_loop_iterations=0,
        candidate_count=0,
        fit_kwargs={},
        seed=0,
    )
    assert (run_root / "experiment1" / "design" / "stimuli.json").exists()


def test_design_resume_reuses_existing_candidates_without_respawning_agent(
    tmp_path, monkeypatch
):
    # experiment1 complete; experiment2 stopped after a design agent left a
    # candidate pool but no stimuli.json. Resume must finish the EIG scoring
    # without re-running the (expensive) design agent.
    run_root = tmp_path / "run"
    _complete_experiment_on_disk(run_root, 1)
    exp2 = run_root / "experiment2"
    exp2_models = exp2 / "cognitive_models"
    exp2_models.mkdir(parents=True)
    for name in ("bayesian_diagnosticity", "encoding_compressibility"):
        shutil.copyfile(SEED_MODELS_DIR / f"{name}.py", exp2_models / f"{name}.py")
    (exp2_models / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [
                {"name": "bayesian_diagnosticity",
                 "rationale": "Fair-coin diagnosticity hypothesis."},
                {"name": "encoding_compressibility",
                 "rationale": "Compressibility-penalty hypothesis."},
            ]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (exp2 / "design").mkdir()
    (exp2 / "design" / "candidates.json").write_text(
        json.dumps(CANDIDATE_POOL), encoding="utf-8"
    )

    spawn_calls = []
    annotate_calls = []
    monkeypatch.setattr(
        holdout_recovery, "spawn_cc_agent", _stub_spawn_cc_agent(spawn_calls)
    )
    monkeypatch.setattr(
        holdout_recovery, "annotate_eig", _stub_annotate_eig(annotate_calls)
    )
    monkeypatch.setattr(
        holdout_recovery, "generate_responses", _stub_generate_responses([])
    )
    monkeypatch.setattr(
        holdout_recovery, "run_inner_model_loop_programmatic", _stub_inner_loop()
    )

    run_holdout_experiments(
        "prototype_similarity",
        PROTOTYPE_GT_PARAMS,
        run_root,
        seed_models_dir=SEED_MODELS_DIR,
        n_experiments=2,
        n_participants=2,
        inner_loop_iterations=0,
        candidate_count=0,
        fit_kwargs={},
        seed=5,
        resume=True,
    )

    # No agent re-run for experiment2 (theory already valid, candidate pool reused).
    assert spawn_calls == []
    assert len(annotate_calls) == 1
    assert (exp2 / "design" / "stimuli.json").exists()


# ── eval pool with post-run exclusion ───────────────────────────────


def _write_training_responses(run_root, exp_num, pairs):
    data_dir = run_root / f"experiment{exp_num}" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    lines = ["participant_id,trial_index,sequence_a,sequence_b,chose_left"]
    lines += [
        f"0,{i},{pair['sequence_a']},{pair['sequence_b']},1"
        for i, pair in enumerate(pairs)
    ]
    (data_dir / "responses.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_eval_stimuli_excludes_trained_pairs_unordered(tmp_path):
    pool = generate_candidate_pool(20, lengths=(6,), seed=11)
    trained = {
        "sequence_a": pool[0]["sequence_b"],  # order flipped vs. the pool entry
        "sequence_b": pool[0]["sequence_a"],
    }
    run_root = tmp_path / "run"
    _write_training_responses(run_root, 1, [trained])

    result = build_eval_stimuli(
        run_root, n_experiments=1, n_pairs=20, lengths=(6,), seed=11, min_remaining=1
    )

    assert result["n_dropped"] == 1
    assert len(result["stimuli"]) == 19
    assert pool[0] not in result["stimuli"]


def test_build_eval_stimuli_raises_when_too_few_remain(tmp_path):
    pool = generate_candidate_pool(5, lengths=(6,), seed=11)
    run_root = tmp_path / "run"
    _write_training_responses(run_root, 1, pool)

    with pytest.raises(ValueError, match="min_remaining"):
        build_eval_stimuli(
            run_root, n_experiments=1, n_pairs=5, lengths=(6,), seed=11,
            min_remaining=5,
        )


def test_collect_trained_pairs_requires_every_responses_csv(tmp_path):
    run_root = tmp_path / "run"
    _write_training_responses(run_root, 1, [DESIGN_STIMULI[0]])
    # experiment2 has no data/responses.csv
    with pytest.raises(FileNotFoundError):
        collect_trained_pairs(run_root, n_experiments=2)


# ── trajectory evaluation ───────────────────────────────────────────

EVAL_STIMULI = [
    {"sequence_a": "HTHTHT", "sequence_b": "HHHHHH"},
    {"sequence_a": "HHHTTT", "sequence_b": "HTHTHT"},
    {"sequence_a": "HHTTHH", "sequence_b": "TTTTTT"},
]


def _write_loop_artifacts(run_root, exp_num, history):
    loop_dir = run_root / f"experiment{exp_num}" / "model_loop"
    (loop_dir / "models").mkdir(parents=True, exist_ok=True)
    (loop_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (loop_dir / "responses.csv").write_text("chose_left\n1\n", encoding="utf-8")


def _history_step(step, iteration, best):
    return {
        "step": step,
        "iteration": iteration,
        "best_model": best,
        "posteriors": {best: 1.0},
        "elpd_loo": {best: -1.0},
    }


def test_evaluate_trajectory_scores_every_history_step(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    _write_loop_artifacts(
        run_root, 1,
        [_history_step(0, None, "model_a"), _history_step(1, 0, "model_b")],
    )
    _write_loop_artifacts(run_root, 2, [_history_step(0, None, "model_b")])

    gt_p = np.array([0.2, 0.5, 0.9])
    predictions = {"model_a": np.array([0.3, 0.9, 0.5]), "model_b": gt_p.copy()}

    monkeypatch.setattr(
        holdout_recovery,
        "p_left_fixed_params",
        lambda model_name, models_dir, stimuli, params, **kw: gt_p,
    )
    monkeypatch.setattr(
        holdout_recovery, "make_stim_data", lambda model, rows: {"n": len(rows)}
    )
    # No model in these stubs indexes a participant random effect.
    monkeypatch.setattr(holdout_recovery, "pm_data_inputs", lambda model: [])

    class Fitted:
        model = None

        def __init__(self, name):
            self.name = name

        def predict_p_left(self, stim_data):
            return predictions[self.name]

    monkeypatch.setattr(
        holdout_recovery,
        "fit_model",
        lambda name, models_dir, responses_path, **kw: Fitted(name),
    )

    rows = evaluate_trajectory(
        run_root,
        "prototype_similarity",
        {"theta_alt": 0.65},
        EVAL_STIMULI,
        seed_models_dir=SEED_MODELS_DIR,
        n_experiments=2,
        cache_dir=None,
        fit_kwargs={},
    )

    assert [(r["experiment"], r["step"], r["global_step"]) for r in rows] == [
        (1, 0, 0),
        (1, 1, 1),
        (2, 0, 2),
    ]
    # model_b predicts the ground truth exactly; model_a does not.
    assert rows[1]["pearson_r"] == pytest.approx(1.0)
    assert rows[1]["rmse"] == pytest.approx(0.0)
    assert rows[0]["pearson_r"] is not None and -1.0 <= rows[0]["pearson_r"] < 1.0
    assert rows[0]["rmse"] > 0.0


def test_evaluate_trajectory_computes_bayesian_model_average(tmp_path, monkeypatch):
    # At a step where the posterior is split across models, the trajectory must
    # report both the single best model's correlation *and* the
    # posterior-weighted Bayesian model average. Zero-weight models never enter
    # the average (and are never fit).
    run_root = tmp_path / "run"
    history = [
        {
            "step": 0,
            "iteration": None,
            "best_model": "model_a",
            "posteriors": {"model_a": 0.75, "model_b": 0.25, "model_c": 0.0},
            "elpd_loo": {"model_a": -1.0, "model_b": -2.0, "model_c": -9.0},
        }
    ]
    _write_loop_artifacts(run_root, 1, history)

    gt_p = np.array([0.2, 0.5, 0.9])
    predictions = {
        "model_a": np.array([0.3, 0.6, 0.8]),
        "model_b": np.array([0.9, 0.1, 0.5]),
        "model_c": np.array([0.0, 0.0, 0.0]),  # zero weight: excluded from BMA
    }
    fit_names = []

    monkeypatch.setattr(
        holdout_recovery,
        "p_left_fixed_params",
        lambda model_name, models_dir, stimuli, params, **kw: gt_p,
    )
    monkeypatch.setattr(
        holdout_recovery, "make_stim_data", lambda model, rows: {"n": len(rows)}
    )
    # No model in these stubs indexes a participant random effect.
    monkeypatch.setattr(holdout_recovery, "pm_data_inputs", lambda model: [])

    class Fitted:
        model = None

        def __init__(self, name):
            self.name = name

        def predict_p_left(self, stim_data):
            return predictions[self.name]

    def fake_fit(name, models_dir, responses_path, **kw):
        fit_names.append(name)
        return Fitted(name)

    monkeypatch.setattr(holdout_recovery, "fit_model", fake_fit)

    rows = evaluate_trajectory(
        run_root,
        "prototype_similarity",
        {"theta_alt": 0.65},
        EVAL_STIMULI,
        seed_models_dir=SEED_MODELS_DIR,
        n_experiments=1,
        cache_dir=None,
        fit_kwargs={},
    )

    row = rows[0]
    # Best line is the single argmax-posterior model.
    assert row["best_model"] == "model_a"
    assert row["pearson_r"] == pytest.approx(pearson_r(gt_p.tolist(), predictions["model_a"].tolist()))
    # BMA line is the posterior-weighted average over nonzero-weight models.
    bma = 0.75 * predictions["model_a"] + 0.25 * predictions["model_b"]
    assert row["pearson_r_bma"] == pytest.approx(pearson_r(gt_p.tolist(), bma.tolist()))
    assert row["rmse_bma"] == pytest.approx(float(np.sqrt(np.mean((gt_p - bma) ** 2))))
    # The zero-weight model is never fit.
    assert "model_c" not in fit_names


def test_evaluate_trajectory_marginalizes_participant_random_effect(
    tmp_path, monkeypatch
):
    # A hierarchical model's p_left is per participant, so it has no
    # population-level prediction of its own. The held-out evaluation must
    # marginalize: replicate each stimulus across the fitted participants and
    # average their p_left. Here the per-participant predictions average exactly
    # to the ground truth, so recovery is perfect.
    run_root = tmp_path / "run"
    loop_dir = run_root / "experiment1" / "model_loop"
    (loop_dir / "models").mkdir(parents=True)
    (loop_dir / "history.json").write_text(
        json.dumps([_history_step(0, None, "hier_model")]), encoding="utf-8"
    )
    (loop_dir / "responses.csv").write_text(
        "participant_id,chose_left\n0,1\n1,0\n", encoding="utf-8"
    )

    eval_stimuli = [
        {"sequence_a": "HTHTHT", "sequence_b": "HHHHHH"},
        {"sequence_a": "HHHTTT", "sequence_b": "HTHTHT"},
    ]
    gt_p = np.array([0.3, 0.7])

    monkeypatch.setattr(
        holdout_recovery, "p_left_fixed_params", lambda *a, **k: gt_p
    )
    monkeypatch.setattr(
        holdout_recovery, "pm_data_inputs",
        lambda model: ["participant_id", "chose_left"],
    )
    # Pass rows straight through so the fake model can read participant_id.
    monkeypatch.setattr(holdout_recovery, "make_stim_data", lambda model, rows: rows)

    # p_left per (participant, stimulus): participant offsets shift the curve.
    table = {
        (0, "HTHTHT"): 0.2, (0, "HHHTTT"): 0.8,
        (1, "HTHTHT"): 0.4, (1, "HHHTTT"): 0.6,
    }

    class Fitted:
        model = None

        def predict_p_left(self, rows):
            return np.array(
                [table[(int(r["participant_id"]), r["sequence_a"])] for r in rows]
            )

    monkeypatch.setattr(
        holdout_recovery, "fit_model",
        lambda name, models_dir, responses_path, **kw: Fitted(),
    )

    rows = evaluate_trajectory(
        run_root,
        "prototype_similarity",
        {"theta_alt": 0.65},
        eval_stimuli,
        seed_models_dir=SEED_MODELS_DIR,
        n_experiments=1,
        cache_dir=None,
        fit_kwargs={},
    )

    # Marginal p_left per stimulus = mean over participants:
    #   HTHTHT: (0.2 + 0.4) / 2 = 0.3 ; HHHTTT: (0.8 + 0.6) / 2 = 0.7 == gt_p.
    assert rows[0]["pearson_r"] == pytest.approx(1.0)
    assert rows[0]["rmse"] == pytest.approx(0.0)
    assert rows[0]["pearson_r_bma"] == pytest.approx(1.0)


def test_fitted_seed_baseline_correlation_pools_all_experiments(tmp_path, monkeypatch):
    # The fitted-seed baseline is one flat number: the other seed models, fit on
    # *all* experiments' pooled responses, correlated with the GT and averaged.
    run_root = tmp_path / "run"
    for exp_num in (1, 2):
        loop_dir = run_root / f"experiment{exp_num}" / "model_loop"
        loop_dir.mkdir(parents=True)
        # Distinct rows per experiment so pooling is observable in the row count.
        (loop_dir / "responses.csv").write_text(
            "chose_left\n1\n0\n" if exp_num == 1 else "chose_left\n1\n",
            encoding="utf-8",
        )

    gt_p = np.array([0.2, 0.5, 0.9])
    predictions = {
        "seed_x": np.array([0.2, 0.5, 0.9]),  # r = +1
        "seed_y": np.array([0.9, 0.6, 0.2]),  # 1.1 - gt_p -> r = -1
    }
    fit_responses = []

    monkeypatch.setattr(
        holdout_recovery, "p_left_fixed_params", lambda *a, **k: gt_p
    )
    monkeypatch.setattr(
        holdout_recovery, "make_stim_data", lambda model, rows: {"n": len(rows)}
    )
    monkeypatch.setattr(holdout_recovery, "pm_data_inputs", lambda model: [])

    class Fitted:
        model = None

        def __init__(self, name):
            self.name = name

        def predict_p_left(self, stim_data):
            return predictions[self.name]

    def fake_fit(name, models_dir, responses_path, **kw):
        fit_responses.append(Path(responses_path).name)
        return Fitted(name)

    monkeypatch.setattr(holdout_recovery, "fit_model", fake_fit)

    out = fitted_seed_baseline_correlation(
        run_root,
        "prototype_similarity",
        {"theta_alt": 0.65},
        EVAL_STIMULI,
        seed_models_dir=SEED_MODELS_DIR,
        n_experiments=2,
        other_seed_models=["seed_x", "seed_y"],
        cache_dir=None,
        fit_kwargs={},
    )

    # Mean of r=+1 (seed_x) and r=-1 (seed_y) is 0.
    assert out["mean_r"] == pytest.approx(0.0)
    assert out["per_model"]["seed_x"]["pearson_r"] == pytest.approx(1.0)
    assert out["per_model"]["seed_y"]["pearson_r"] == pytest.approx(-1.0)
    # Both seeds were fit on the single pooled CSV (3 rows = 2 from exp1 + 1 exp2).
    assert fit_responses == ["pooled_responses.csv", "pooled_responses.csv"]
    assert out["n_responses"] == 3


def test_seed_baseline_correlation_averages_other_seed_models(monkeypatch):
    # The no-learning baseline averages the correlation of every seed model
    # *except* the ground truth (each with its default params) against the GT.
    gt_p = np.array([0.2, 0.5, 0.9])
    preds = {
        "gt": gt_p,
        "other_a": np.array([0.2, 0.5, 0.9]),  # perfectly correlated -> r = 1
        "other_b": np.array([0.9, 0.6, 0.2]),  # 1.1 - gt_p, exact anti -> r = -1
    }
    monkeypatch.setattr(
        holdout_recovery,
        "p_left_fixed_params",
        lambda model_name, models_dir, stimuli, params, **kw: preds[model_name],
    )
    monkeypatch.setattr(
        holdout_recovery,
        "resolve_generating_params",
        lambda spec, seed_models_dir: {
            "gt": {"a": 1.0}, "other_a": {"a": 1.0}, "other_b": {"a": 1.0}
        },
    )

    out = seed_baseline_correlation(
        "gt", {"a": 1.0}, EVAL_STIMULI, seed_models_dir=SEED_MODELS_DIR
    )

    assert set(out["per_model"]) == {"other_a", "other_b"}  # GT excluded
    assert out["per_model"]["other_a"] == pytest.approx(1.0)
    assert out["per_model"]["other_b"] == pytest.approx(-1.0)
    assert out["mean_r"] == pytest.approx(0.0)


def test_reevaluate_trajectories_recomputes_best_and_bma_from_disk(tmp_path, monkeypatch):
    # Regenerating metrics for a finished run reads its on-disk history and
    # eval_stimuli, recomputes both trajectories through the cache, and returns
    # a *copy* (the input result is left untouched).
    run_root = tmp_path / "runs" / "prototype_similarity"
    _write_loop_artifacts(run_root, 1, [_history_step(0, None, "model_a")])
    (run_root / "eval_stimuli.json").write_text(
        json.dumps(EVAL_STIMULI), encoding="utf-8"
    )

    gt_p = np.array([0.2, 0.5, 0.9])
    monkeypatch.setattr(
        holdout_recovery,
        "p_left_fixed_params",
        lambda model_name, models_dir, stimuli, params, **kw: gt_p,
    )
    monkeypatch.setattr(
        holdout_recovery, "make_stim_data", lambda model, rows: {"n": len(rows)}
    )
    # No model in these stubs indexes a participant random effect.
    monkeypatch.setattr(holdout_recovery, "pm_data_inputs", lambda model: [])

    class Fitted:
        model = None

        def predict_p_left(self, stim_data):
            return gt_p

    monkeypatch.setattr(
        holdout_recovery, "fit_model",
        lambda name, models_dir, responses_path, **kw: Fitted(),
    )

    result = {
        "n_experiments": 1,
        "fit_kwargs": {},
        "seed_models_dir": str(SEED_MODELS_DIR),
        "gt_runs": [
            {
                "gt_model": "prototype_similarity",
                "params": {"theta_alt": 0.65},
                "run_root": str(run_root),
                "trajectory": [{"placeholder": True}],
            }
        ],
    }

    enriched = reevaluate_trajectories(
        result, seed_models_dir=SEED_MODELS_DIR, cache_dir=None
    )

    traj = enriched["gt_runs"][0]["trajectory"]
    assert len(traj) == 1
    assert traj[0]["pearson_r"] == pytest.approx(1.0)
    assert traj[0]["pearson_r_bma"] == pytest.approx(1.0)
    # Both seed-model baselines are attached at the gt_run level (flat numbers).
    assert enriched["gt_runs"][0]["fitted_baseline"]["mean_r"] == pytest.approx(1.0)
    # The no-learning baseline is attached (other seeds vs. GT, all stubbed equal).
    baseline = enriched["gt_runs"][0]["baseline"]
    assert set(baseline["per_model"]) == {
        "bayesian_diagnosticity",
        "encoding_compressibility",
        "window_typicality",
    }
    assert baseline["mean_r"] == pytest.approx(1.0)
    # The original result is not mutated.
    assert result["gt_runs"][0]["trajectory"] == [{"placeholder": True}]


def test_evaluate_trajectory_fails_loudly_without_history(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    (run_root / "experiment1" / "model_loop").mkdir(parents=True)
    monkeypatch.setattr(
        holdout_recovery,
        "p_left_fixed_params",
        lambda model_name, models_dir, stimuli, params, **kw: np.zeros(3),
    )
    with pytest.raises(FileNotFoundError, match="history.json"):
        evaluate_trajectory(
            run_root,
            "prototype_similarity",
            {},
            EVAL_STIMULI,
            seed_models_dir=SEED_MODELS_DIR,
            n_experiments=1,
            cache_dir=None,
            fit_kwargs={},
        )


# ── leakage check ───────────────────────────────────────────────────


def _make_model_dirs(run_root, exp_num, files):
    models_dir = run_root / f"experiment{exp_num}" / "cognitive_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (models_dir / name).write_text(content, encoding="utf-8")


def test_leakage_check_flags_identical_file(tmp_path):
    gt_source = (SEED_MODELS_DIR / "prototype_similarity.py").read_text(
        encoding="utf-8"
    )
    run_root = tmp_path / "run"
    _make_model_dirs(run_root, 1, {"sneaky_copy.py": gt_source})

    result = leakage_check(
        run_root, "prototype_similarity", seed_models_dir=SEED_MODELS_DIR,
        n_experiments=1,
    )
    assert result["any_identical"] is True
    flagged = [f for f in result["files"] if f["identical"]]
    assert flagged and flagged[0]["path"].endswith("sneaky_copy.py")


def test_leakage_check_flags_distinctive_param_mentions(tmp_path):
    run_root = tmp_path / "run"
    _make_model_dirs(
        run_root, 1, {"candidate.py": "# uses theta_alt as a parameter\n"}
    )

    result = leakage_check(
        run_root, "prototype_similarity", seed_models_dir=SEED_MODELS_DIR,
        n_experiments=1,
    )
    assert result["any_mention"] is True
    assert result["any_identical"] is False


def test_leakage_check_flags_gt_named_file(tmp_path):
    run_root = tmp_path / "run"
    _make_model_dirs(run_root, 1, {"prototype_similarity.py": "# innocuous body\n"})

    result = leakage_check(
        run_root, "prototype_similarity", seed_models_dir=SEED_MODELS_DIR,
        n_experiments=1,
    )
    assert result["any_gt_named"] is True
    assert result["any_identical"] is False


def test_leakage_check_clean_run_unflagged(tmp_path):
    run_root = tmp_path / "run"
    # beta/side_bias are shared across all families and must not trip the flag.
    _make_model_dirs(run_root, 1, {"candidate.py": "# beta and side_bias only\n"})

    result = leakage_check(
        run_root, "prototype_similarity", seed_models_dir=SEED_MODELS_DIR,
        n_experiments=1,
    )
    assert result["any_identical"] is False
    assert result["any_mention"] is False
    assert result["any_gt_named"] is False


# ── config-bridge guards ────────────────────────────────────────────


def test_from_config_rejects_foreign_seed_models_dir(tmp_path):
    # Experiment 1 is seeded from the project assets, so a config pointing the
    # generator at any other seed directory would be incoherent.
    config = {
        "seed_models_dir": str(tmp_path / "other_seeds"),
        "gt_models": ["prototype_similarity"],
    }
    with pytest.raises(ValueError, match="project's seed"):
        run_holdout_recovery_from_config(
            config, tmp_path / "config.yaml", tmp_path / "runs"
        )


def test_from_config_rejects_unknown_gt_model_override(tmp_path):
    config = {
        "seed_models_dir": str(SEED_MODELS_DIR),
        "gt_models": ["prototype_similarity"],
    }
    with pytest.raises(ValueError, match="not among the configured"):
        run_holdout_recovery_from_config(
            config,
            tmp_path / "config.yaml",
            tmp_path / "runs",
            gt_model_override="encoding_compressibility",
        )


def test_from_config_rejects_zero_overrides(tmp_path, monkeypatch):
    # An explicit 0 must fail loudly, not be silently swallowed by a falsy-`or`
    # fallback to the config default. The tripwire stub guarantees the guard
    # fires before any experiment work (a real agent spawn) could start.
    def tripwire(*args, **kwargs):
        raise AssertionError("guard did not fire before the experiment loop")

    monkeypatch.setattr(holdout_recovery, "spawn_cc_agent", tripwire)
    monkeypatch.setattr(
        holdout_recovery, "seed_experiment_models_from_project", tripwire
    )

    config = {
        "seed_models_dir": str(SEED_MODELS_DIR),
        "gt_models": ["prototype_similarity"],
    }
    with pytest.raises(ValueError, match="n_experiments"):
        run_holdout_recovery_from_config(
            config, tmp_path / "config.yaml", tmp_path / "runs",
            n_experiments_override=0,
        )
    with pytest.raises(ValueError, match="n_participants"):
        run_holdout_recovery_from_config(
            config, tmp_path / "config.yaml", tmp_path / "runs",
            n_participants_override=0,
        )


# ── tidy rows ───────────────────────────────────────────────────────


def test_trajectory_tidy_rows_one_row_per_step():
    result = {
        "gt_runs": [
            {
                "gt_model": "prototype_similarity",
                "trajectory": [
                    {"experiment": 1, "step": 0, "iteration": None,
                     "global_step": 0, "best_model": "a", "pearson_r": 0.5,
                     "rmse": 0.1, "pearson_r_bma": 0.6, "rmse_bma": 0.08},
                    {"experiment": 1, "step": 1, "iteration": 0,
                     "global_step": 1, "best_model": "b", "pearson_r": None,
                     "rmse": 0.2, "pearson_r_bma": None, "rmse_bma": 0.2},
                ],
            }
        ]
    }
    rows = trajectory_tidy_rows(result)
    assert len(rows) == 2
    assert all(set(TRAJECTORY_COLUMNS) <= set(row) for row in rows)
    assert rows[0]["gt_model"] == "prototype_similarity"
    assert rows[1]["pearson_r"] is None


# ── plot ────────────────────────────────────────────────────────────


def test_plot_holdout_trajectories_writes_png(tmp_path):
    from src.subjective_randomness.reporting import plot_holdout_trajectories

    result = {
        "gt_runs": [
            {
                "gt_model": "prototype_similarity",
                "baseline": {"mean_r": 0.55, "per_model": {"a": 0.5, "b": 0.6}},
                "fitted_baseline": {"mean_r": 0.8, "mean_rmse": 0.07,
                                    "per_model": {}, "n_responses": 100},
                "trajectory": [
                    {"experiment": 1, "step": 0, "iteration": None,
                     "global_step": 0, "best_model": "a", "pearson_r": 0.4,
                     "rmse": 0.2, "pearson_r_bma": 0.5, "rmse_bma": 0.18},
                    {"experiment": 2, "step": 0, "iteration": None,
                     "global_step": 1, "best_model": "a", "pearson_r": None,
                     "rmse": 0.3, "pearson_r_bma": None, "rmse_bma": 0.3},
                    {"experiment": 2, "step": 1, "iteration": 0,
                     "global_step": 2, "best_model": "b", "pearson_r": 0.9,
                     "rmse": 0.05, "pearson_r_bma": 0.95, "rmse_bma": 0.03},
                ],
            },
            {
                "gt_model": "encoding_compressibility",
                # No baseline keys: the plot must tolerate their absence.
                "trajectory": [
                    {"experiment": 1, "step": 0, "iteration": None,
                     "global_step": 0, "best_model": "c", "pearson_r": 0.7,
                     "rmse": 0.1, "pearson_r_bma": 0.72, "rmse_bma": 0.09},
                ],
            },
        ]
    }
    out_path = tmp_path / "figs" / "holdout.png"
    plot_holdout_trajectories(result, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # The same result renders an RMSE figure from the rmse/rmse_bma keys and the
    # fitted-seed mean_rmse baseline; the default-params baseline (mean_r only)
    # is absent here and must be tolerated.
    rmse_path = tmp_path / "figs" / "holdout_rmse.png"
    plot_holdout_trajectories(result, rmse_path, metric="rmse")
    assert rmse_path.exists()
    assert rmse_path.stat().st_size > 0


def test_plot_holdout_trajectories_rejects_unknown_metric(tmp_path):
    from src.subjective_randomness.reporting import plot_holdout_trajectories

    with pytest.raises(ValueError, match="Unknown metric"):
        plot_holdout_trajectories({"gt_runs": []}, tmp_path / "x.png", metric="mae")


# ── CLI parsing ─────────────────────────────────────────────────────


def _load_cli_script():
    import importlib.util
    import sys as _sys

    path = REPO_ROOT / "scripts" / "subjective_randomness" / "holdout_recovery.py"
    spec = importlib.util.spec_from_file_location("_sr_script_holdout_recovery", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the @dataclass Args can resolve its own module.
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_holdout_cli_defaults_and_overrides():
    import tyro

    args_cls = _load_cli_script().Args

    default = tyro.cli(args_cls, args=["--config", "c.yaml", "--out", "h.json"])
    assert default.config == Path("c.yaml")
    assert default.out == Path("h.json")
    assert default.tidy_csv is None
    assert default.figure is None
    assert default.results_root is None
    assert default.cache_dir is None
    assert default.gt_model is None
    assert default.n_experiments is None
    assert default.draws is None  # falls back to the config's fit settings
    assert default.backend is None
    assert default.resume is False

    full = tyro.cli(
        args_cls,
        args=[
            "--config", "c.yaml",
            "--out", "h.json",
            "--tidy-csv", "h.csv",
            "--figure", "h.png",
            "--gt-model", "prototype_similarity",
            "--n-experiments", "2",
            "--n-participants", "10",
            "--inner-loop-iterations", "1",
            "--inner-loop-candidates", "2",
            "--draws", "100",
            "--tune", "100",
            "--chains", "2",
            "--seed", "3",
            "--agent-timeout-sec", "300",
            "--backend", "claude",
            "--resume",
        ],
    )
    assert full.resume is True
    assert full.gt_model == "prototype_similarity"
    assert full.n_experiments == 2
    assert full.inner_loop_iterations == 1
    assert full.inner_loop_candidates == 2
    assert full.draws == 100
    assert full.agent_timeout_sec == 300
    assert full.backend == "claude"
    assert full.figure == Path("h.png")


def _load_plot_cli_script():
    import importlib.util
    import sys as _sys

    path = REPO_ROOT / "scripts" / "subjective_randomness" / "plot_holdout_recovery.py"
    spec = importlib.util.spec_from_file_location("_sr_script_plot_holdout", path)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_plot_holdout_cli_defaults_derive_from_result_path():
    import tyro

    args_cls = _load_plot_cli_script().Args

    default = tyro.cli(args_cls, args=["--result", "x/holdout.json"])
    assert default.result == Path("x/holdout.json")
    # Figure, tidy CSV, and enriched JSON default alongside the result file.
    assert default.figure is None
    assert default.tidy_csv is None
    assert default.out is None
    assert default.cache_dir is None
    assert default.seed_models_dir is None

    full = tyro.cli(
        args_cls,
        args=[
            "--result", "x/holdout.json",
            "--figure", "x/holdout.png",
            "--tidy-csv", "x/holdout.csv",
            "--out", "x/holdout.json",
            "--cache-dir", "x/mcmc_cache",
            "--seed-models-dir", "seeds",
        ],
    )
    assert full.figure == Path("x/holdout.png")
    assert full.tidy_csv == Path("x/holdout.csv")
    assert full.cache_dir == Path("x/mcmc_cache")
    assert full.seed_models_dir == Path("seeds")


# ── real-MCMC proof (slow) ──────────────────────────────────────────


@pytest.mark.slow
def test_holdout_single_experiment_real_mcmc_with_stub_agents(tmp_path, monkeypatch):
    """End-to-end with real MCMC: only the coding agents are stubbed.

    Proves the history.json -> cached fit -> posterior-predictive chain works
    with real PyMC: the run's fits land in the shared cache and the trajectory
    evaluation reuses them without sampling anew.
    """
    design_stimuli = [
        {"sequence_a": "HTHTHT", "sequence_b": "HHHHHH", "eig": 0.9},
        {"sequence_a": "HHHTTT", "sequence_b": "HTHTHT", "eig": 0.7},
        {"sequence_a": "HHTTHH", "sequence_b": "TTTTTT", "eig": 0.5},
        {"sequence_a": "THTHTH", "sequence_b": "TTTHHH", "eig": 0.4},
    ]

    def design_only_spawn(agent_key, exp_dir, **kwargs):
        assert agent_key == "2_design"  # single experiment: no theory agent
        design_dir = exp_dir / "design"
        design_dir.mkdir(parents=True, exist_ok=True)
        (design_dir / "stimuli.json").write_text(
            json.dumps(design_stimuli), encoding="utf-8"
        )
        return True, "ok"

    monkeypatch.setattr(holdout_recovery, "spawn_cc_agent", design_only_spawn)

    config = {
        "project_id": "subjective_randomness",
        "seed_models_dir": str(SEED_MODELS_DIR),
        "gt_models": ["prototype_similarity"],
        "n_experiments": 1,
        "n_participants": 8,
        "seed": 3,
        "inner_loop": {"max_iterations": 0, "candidate_count": 0},
        "eval_pool": {"n_pairs": 30, "lengths": [6], "seed": 11, "min_remaining": 5},
        "fit": {"draws": 150, "tune": 150, "chains": 2},
    }
    cache_dir = tmp_path / "cache"

    result = run_holdout_recovery_from_config(
        config, tmp_path / "config.yaml", tmp_path / "runs", cache_dir=cache_dir
    )

    gt_run = result["gt_runs"][0]
    trajectory = gt_run["trajectory"]
    assert len(trajectory) == 1  # seed-only scoring step
    # The held-out GT (prototype_similarity) is excluded from the seed set; the
    # recovered best model is whichever of the remaining seeds best fits the
    # small design sample. This is a pipeline/caching smoke test, so we only
    # require a valid seeded model, not a specific winner.
    assert trajectory[0]["best_model"] in {
        "bayesian_diagnosticity",
        "encoding_compressibility",
        "window_typicality",
    }
    r = trajectory[0]["pearson_r"]
    assert r is not None and -1.0 <= r <= 1.0
    assert trajectory[0]["rmse"] >= 0.0

    # The run's MCMC fits landed in the shared cache...
    cached = {p.name for p in cache_dir.glob("*.nc")}
    assert cached
    # ...and re-evaluating the trajectory is a pure cache hit: same numbers,
    # no new fit files.
    run_root = tmp_path / "runs" / "prototype_similarity"
    eval_stimuli = json.loads(
        (run_root / "eval_stimuli.json").read_text(encoding="utf-8")
    )
    rows = evaluate_trajectory(
        run_root,
        "prototype_similarity",
        gt_run["params"],
        eval_stimuli,
        seed_models_dir=SEED_MODELS_DIR,
        n_experiments=1,
        cache_dir=cache_dir,
        fit_kwargs=config["fit"],
    )
    assert {p.name for p in cache_dir.glob("*.nc")} == cached
    assert rows[0]["pearson_r"] == pytest.approx(r)
