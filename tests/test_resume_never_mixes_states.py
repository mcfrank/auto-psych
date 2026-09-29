"""A resume must find a stage complete or redo it from scratch (first audit R2;
second audit B14).

Crashes are simulated at each write of the boundary between experiments:

- **carry-forward** (and experiment 1's seeding) used to copy the model files,
  then the manifest, then the ledger into ``cognitive_models/``. A crash after
  the manifest left a valid-looking set with no ledger, and the manifest's
  existence made the copy count as done forever. The set is now built in a
  temporary directory and renamed into place: a crash leaves the old state
  (no set) or the new one (files, manifest and ledger together).
- **export + registry**: ``cognitive_models/`` is both the model loop's input
  and its export, and the registry update ran after the export, outside any
  validation. A crash in between let a resume skip the stage (its validator
  only asked whether the best model was in the set) or rerun the loop seeded
  from its own half-written export. The stage now starts by recording its
  input set (``cognitive_models_input/``) and the run's agent notes
  (``agent_notes_at_start/``), or restoring them when a previous attempt
  recorded them, wiping ``model_loop/`` and resetting the registry; it ends by
  writing the registry and then ``model_loop/export_complete.json``, which the
  validator checks against the set and the registry.
- **agent notes** written during an abandoned attempt are discarded: the notes
  go back to what they were when the stage first started.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from src.models.model_manifest import read_manifest_names
from src.pipelines.outer_loop import model_loop_runner as mlr
from src.pipelines.outer_loop.orchestrator import (
    carry_forward_cognitive_models,
    seed_experiment_models_from_project,
)
from src.pipelines.outer_loop.orchestrator_validators import validate_cc_output
from src.registry.io import get_model_weights

LEDGER = "attempted_hypotheses.jsonl"


class Crash(RuntimeError):
    """A simulated kill of the process at a chosen point."""


def _write_set(models_dir: Path, names, *, ledger_lines=("{}",)) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (models_dir / f"{name}.py").write_text(f"# {name}\n", encoding="utf-8")
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": n, "rationale": f"{n} hypothesis"} for n in names]}
        ),
        encoding="utf-8",
    )
    if ledger_lines is not None:
        (models_dir / LEDGER).write_text(
            "\n".join(ledger_lines) + "\n", encoding="utf-8"
        )


def _crash_on_copy_of(monkeypatch, filename: str) -> None:
    real_copy = shutil.copyfile

    def copy(src, dst, *args, **kwargs):
        if Path(dst).name == filename:
            raise Crash(f"killed while copying {filename}")
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", copy)


# ── carry-forward and seeding ────────────────────────────────────────────


@pytest.mark.parametrize("killed_at", ["b.py", "models_manifest.yaml", LEDGER])
def test_a_crashed_carry_forward_leaves_no_set_and_reruns_whole(
    tmp_path, monkeypatch, killed_at
):
    prev, new = tmp_path / "experiment1", tmp_path / "experiment2"
    _write_set(prev / "cognitive_models", ["a", "b"], ledger_lines=['{"x": 1}'])
    (new / "cognitive_models").mkdir(parents=True)  # ensure_experiment_dirs made it

    with monkeypatch.context() as m:
        _crash_on_copy_of(m, killed_at)
        with pytest.raises(Crash):
            carry_forward_cognitive_models(prev, new)
    # The old state: no manifest, so the stage is not done.
    assert not (new / "cognitive_models" / "models_manifest.yaml").exists()
    assert not validate_cc_output("models", new)[0]

    assert carry_forward_cognitive_models(prev, new)
    dest = new / "cognitive_models"
    assert read_manifest_names(dest) == ["a", "b"]
    assert (dest / LEDGER).read_text(encoding="utf-8") == '{"x": 1}\n'
    assert not list(tmp_path.glob("experiment2/.*partial*"))


def test_a_crashed_seeding_leaves_no_set(tmp_path, monkeypatch):
    seed_dir = tmp_path / "seeds"
    _write_set(seed_dir, ["s1", "s2", "gt"], ledger_lines=None)
    exp = tmp_path / "experiment1"
    (exp / "cognitive_models").mkdir(parents=True)
    with monkeypatch.context() as m:
        _crash_on_copy_of(m, "s2.py")
        with pytest.raises(Crash):
            seed_experiment_models_from_project(
                exp, "unused", exclude=["gt"], seed_dir=seed_dir
            )
    assert not (exp / "cognitive_models" / "models_manifest.yaml").exists()
    assert seed_experiment_models_from_project(
        exp, "unused", exclude=["gt"], seed_dir=seed_dir
    )
    assert read_manifest_names(exp / "cognitive_models") == ["s1", "s2"]


# ── the model-loop stage: input, notes, export and registry ─────────────


def _experiment(tmp_path: Path) -> Path:
    exp = tmp_path / "run" / "experiment2"
    for sub in ("cognitive_models", "design", "data", "model_loop"):
        (exp / sub).mkdir(parents=True, exist_ok=True)
    _write_set(
        exp / "cognitive_models", ["seed_a", "carried_b"], ledger_lines=['{"old": 1}']
    )
    mlr.init_registry(exp)
    notes = mlr.agent_notes_dir(exp)
    notes.mkdir(parents=True)
    (notes / "MEMORY.md").write_text("notes from experiment 1\n", encoding="utf-8")
    return exp


def _fake_loop_and_export(exp: Path, *, best="new_c") -> None:
    """What an attempt of the loop leaves: a zoo, a report, notes, and an export
    over ``cognitive_models/`` (carried_b pruned, new_c added)."""
    loop = exp / "model_loop"
    (loop / "models").mkdir(parents=True, exist_ok=True)
    (loop / "model_posterior.json").write_text(
        json.dumps(
            {"posteriors": {best: 1.0}, "best_model": best, "comparison": {best: {}}}
        ),
        encoding="utf-8",
    )
    (loop / "report.md").write_text("# report\n", encoding="utf-8")
    (loop / "history.json").write_text("[]", encoding="utf-8")
    notes = mlr.agent_notes_dir(exp)
    (notes / "MEMORY.md").write_text(
        "notes about candidates of the attempt\n", encoding="utf-8"
    )
    (notes / "attempt.md").write_text("more\n", encoding="utf-8")
    _write_set(
        exp / "cognitive_models",
        ["seed_a", best],
        ledger_lines=['{"old": 1}', '{"new": 2}'],
    )


def test_the_stage_records_its_input_and_notes_on_first_start(tmp_path):
    exp = _experiment(tmp_path)
    mlr.begin_model_loop_stage(exp)
    assert read_manifest_names(exp / mlr.MODEL_LOOP_INPUT_DIRNAME) == [
        "seed_a",
        "carried_b",
    ]
    assert (exp / mlr.AGENT_NOTES_SNAPSHOT_DIRNAME / "MEMORY.md").read_text(
        encoding="utf-8"
    ) == "notes from experiment 1\n"


@pytest.mark.parametrize("crash_after", ["export", "registry"])
def test_a_restart_after_a_crash_redoes_the_stage_from_its_recorded_input(
    tmp_path, crash_after
):
    exp = _experiment(tmp_path)
    mlr.begin_model_loop_stage(exp)
    _fake_loop_and_export(exp)
    if crash_after == "registry":
        mlr.update_registry_from_interpretation(exp)
    # Killed before the completion record: the stage is not done.
    assert not validate_cc_output("5_model_loop", exp)[0]

    mlr.begin_model_loop_stage(exp)

    models = exp / "cognitive_models"
    assert read_manifest_names(models) == ["seed_a", "carried_b"]
    assert sorted(p.name for p in models.glob("*.py")) == ["carried_b.py", "seed_a.py"]
    assert (models / LEDGER).read_text(encoding="utf-8") == '{"old": 1}\n'
    assert list((exp / "model_loop").iterdir()) == []
    assert get_model_weights(exp / "model_registry.yaml") == {}
    notes = mlr.agent_notes_dir(exp)
    assert sorted(p.name for p in notes.iterdir()) == ["MEMORY.md"]
    assert (notes / "MEMORY.md").read_text(
        encoding="utf-8"
    ) == "notes from experiment 1\n"


def test_a_crash_while_restoring_the_input_is_itself_recoverable(tmp_path, monkeypatch):
    exp = _experiment(tmp_path)
    mlr.begin_model_loop_stage(exp)
    _fake_loop_and_export(exp)
    with monkeypatch.context() as m:
        _crash_on_copy_of(m, "carried_b.py")
        with pytest.raises(Crash):
            mlr.begin_model_loop_stage(exp)
    mlr.begin_model_loop_stage(exp)
    assert read_manifest_names(exp / "cognitive_models") == ["seed_a", "carried_b"]


def test_a_finished_stage_validates_and_records_what_it_exported(tmp_path):
    exp = _experiment(tmp_path)
    mlr.begin_model_loop_stage(exp)
    _fake_loop_and_export(exp)
    mlr.finish_model_loop_stage(exp)

    assert validate_cc_output("5_model_loop", exp)[0]
    record = json.loads((exp / "model_loop" / mlr.EXPORT_RECORD_FILENAME).read_text())
    assert record["models"] == ["seed_a", "new_c"]
    assert get_model_weights(exp / "model_registry.yaml") == {
        "seed_a": 0.5,
        "new_c": 0.5,
    }


@pytest.mark.parametrize("tamper", ["manifest", "registry", "ledger", "file"])
def test_the_validator_refuses_an_export_that_disagrees_with_its_record(
    tmp_path, tamper
):
    exp = _experiment(tmp_path)
    mlr.begin_model_loop_stage(exp)
    _fake_loop_and_export(exp)
    mlr.finish_model_loop_stage(exp)
    models = exp / "cognitive_models"
    if tamper == "manifest":
        _write_set(models, ["seed_a"], ledger_lines=['{"old": 1}', '{"new": 2}'])
    elif tamper == "registry":
        from src.registry.io import write_registry

        write_registry(
            exp / "model_registry.yaml", {"seed_a": 1.0}, reserved_for_new=0.0
        )
    elif tamper == "ledger":
        (models / LEDGER).unlink()
    else:
        (models / "new_c.py").unlink()
    ok, message = validate_cc_output("5_model_loop", exp)
    assert not ok, message


def test_a_model_loop_left_by_code_without_the_record_raises(tmp_path):
    # Without the recorded input, cognitive_models/ may already be an export:
    # the stage cannot tell, so it refuses rather than seeding from it.
    exp = _experiment(tmp_path)
    (exp / "model_loop" / "report.md").write_text("# old\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=mlr.MODEL_LOOP_INPUT_DIRNAME):
        mlr.begin_model_loop_stage(exp)


def test_the_design_reads_the_recorded_input_set(tmp_path):
    exp = _experiment(tmp_path)
    assert mlr.experiment_input_models_dir(exp) == exp / "cognitive_models"
    mlr.begin_model_loop_stage(exp)
    _fake_loop_and_export(exp)
    assert mlr.experiment_input_models_dir(exp) == exp / mlr.MODEL_LOOP_INPUT_DIRNAME


# ── the holdout harness, end to end ──────────────────────────────────────


def test_a_cell_killed_between_export_and_registry_redoes_its_model_loop(
    tmp_path, monkeypatch
):
    """The harness resumes an experiment whose loop exported and was killed
    before the registry: the rerun loop starts from the carried set (not the
    export), with the notes as they were, and the stage then completes."""
    from src.subjective_randomness import holdout_recovery
    from tests.test_subjective_randomness_holdout_recovery import (
        SEED_MODELS_DIR,
        _complete_experiment_on_disk,
        _stub_design,
        _stub_generate_responses,
        _stub_inner_loop,
    )

    run_root = tmp_path / "run"
    exp_dir = _complete_experiment_on_disk(run_root, 1, with_model_loop=False)
    started_with = read_manifest_names(exp_dir / "cognitive_models")
    stub = _stub_inner_loop("falk_konold_dp")
    seen = []

    def exporting_loop(exp_dir, **kwargs):
        seen.append(
            {
                "set": read_manifest_names(exp_dir / "cognitive_models"),
                "notes": sorted(p.name for p in mlr.agent_notes_dir(exp_dir).glob("*")),
            }
        )
        loop_dir = stub(exp_dir, **kwargs)
        notes = mlr.agent_notes_dir(exp_dir)
        notes.mkdir(exist_ok=True)
        (notes / f"attempt{len(seen)}.md").write_text("notes\n", encoding="utf-8")
        # The export: a new model joins the carried set.
        models = exp_dir / "cognitive_models"
        shutil.copyfile(models / "falk_konold_dp.py", models / "new_model.py")
        names = read_manifest_names(models) + ["new_model"]
        (models / "models_manifest.yaml").write_text(
            yaml.safe_dump({"models": [{"name": n, "rationale": n} for n in names]}),
            encoding="utf-8",
        )
        return loop_dir

    monkeypatch.setattr(
        holdout_recovery, "run_inner_model_loop_programmatic", exporting_loop
    )
    monkeypatch.setattr(holdout_recovery, "run_design_programmatic", _stub_design([]))
    monkeypatch.setattr(
        holdout_recovery, "generate_responses", _stub_generate_responses([])
    )
    real_finish = mlr.finish_model_loop_stage

    def killed(exp_dir):
        mlr.update_registry_from_interpretation(exp_dir)
        raise Crash("killed before the export record")

    run = dict(
        gt_model="local_representativeness",
        gt_params={
            "theta_alt": 0.65,
            "alt_weight": 0.55,
            "beta": 4.0,
            "side_bias": 0.0,
        },
        run_root=run_root,
        seed_models_dir=SEED_MODELS_DIR,
        n_experiments=1,
        n_participants=2,
        inner_loop_iterations=0,
        candidate_count=0,
        fit_kwargs={},
        seed=0,
        resume=True,
    )
    monkeypatch.setattr(holdout_recovery, "finish_model_loop_stage", killed)
    with pytest.raises(Crash):
        holdout_recovery.run_holdout_experiments(**run)
    assert not validate_cc_output("5_model_loop", exp_dir)[0]

    monkeypatch.setattr(holdout_recovery, "finish_model_loop_stage", real_finish)
    holdout_recovery.run_holdout_experiments(**run)

    assert [s["set"] for s in seen] == [started_with, started_with]
    assert [s["notes"] for s in seen] == [[], []]
    assert validate_cc_output("5_model_loop", exp_dir)[0]
    assert read_manifest_names(exp_dir / "cognitive_models") == started_with + [
        "new_model"
    ]
    assert sorted(get_model_weights(exp_dir / "model_registry.yaml")) == sorted(
        started_with + ["new_model"]
    )


def test_run_py_carries_forward_only_a_finished_model_loop(tmp_path, monkeypatch):
    from src.pipelines.outer_loop import run as outer_run

    monkeypatch.setenv("AUTO_PSYCH_OUTPUT_DIR", str(tmp_path / "output"))
    exp1 = outer_run.experiment_dir("subjective_randomness", 1)
    _write_set(exp1 / "cognitive_models", ["seed_a"])
    exp2 = outer_run.experiment_dir("subjective_randomness", 2)
    exp2.mkdir(parents=True)
    stages = dict(
        project_id="subjective_randomness",
        exp_num=2,
        exp_dir_path=exp2,
        mode="simulated_participants_nobrowser",
        n_participants=2,
        validate=True,
        ground_truth_model=None,
        agent_filter=None,
        inner_loop_iterations=0,
        inner_loop_candidates=0,
        fit_kwargs=None,
        backend=None,
        participant_backend="gemini",
        participant_model=None,
        deploy_target="none",
        collection_owner="x",
        firebase_project=None,
        firebase_region="x",
        prolific_mode="none",
        deploy_only=False,
        prepare_smoke_experiment=False,
        enable_critique=False,
        n_critique_proposals=None,
        critique_alpha=None,
        run_label=None,
        max_validation_repairs=0,
        candidate_hints=None,
        novelty_rmse_threshold=None,
        prune_dse_multiplier=None,
        candidate_parallelism=None,
        publish_another_prolific_study=False,
    )
    with pytest.raises(SystemExit):
        outer_run._run_experiment_stages(**stages)
    assert not (exp2 / "cognitive_models").exists()
