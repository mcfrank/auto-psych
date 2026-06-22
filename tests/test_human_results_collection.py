"""Collecting a finished live (human) outer-loop run into the repo must copy each
real run's experiment tree (responses, design, cognitive models, model-loop
results, agent logs) while:

* excluding the giant MCMC ``.nc`` fit-caches and heavy worktrees / node_modules,
* SCRUBBING every Prolific ID — the raw worker IDs live in the
  ``participant_id_str`` column of responses.csv and, as 24-hex tokens, are also
  sprinkled through logs, deployment manifests, configs and agent transcripts, and
* rendering a SUMMARY.md that, per (run, experiment), reports the winning model,
  its posterior, and the ELPD margin over the runner-up.

Pilots and validation runs are excluded by default. If any Prolific ID would
survive into the repo, collection fails loudly rather than leaking it.
"""

from __future__ import annotations

import json

import pytest

from src.pipelines.outer_loop.results_collection import (
    collect_human_results,
    discover_runs,
    find_prolific_ids,
    redact_csv_text,
    redact_text,
    render_human_experiment_summary,
    summarize_model_posterior,
)

PROJECT = "subjective_randomness"
FAKE_PID = "56fc4404f758b80010affdbb"  # 24-hex Prolific worker id
FAKE_STUDY_ID = "6a385696055b5982b2ed47c7"  # 24-hex Prolific study id


def _posterior_fixture() -> dict:
    return {
        "posteriors": {
            "prototype_similarity": 0.0,
            "iter0_candidate0": 0.000795,
            "iter1_candidate0": 0.999205,
        },
        "n_trials": 1280,
        "comparison": {
            "iter1_candidate0": {"rank": 0, "elpd_loo": -788.83, "elpd_diff": 0.0, "dse": 0.0},
            "iter0_candidate0": {"rank": 1, "elpd_loo": -796.31, "elpd_diff": 7.486, "dse": 4.011},
            "prototype_similarity": {"rank": 2, "elpd_loo": -809.56, "elpd_diff": 20.73, "dse": 5.59},
        },
    }


def _write_experiment(exp_dir, posterior=None):
    """Write a minimal experiment tree, with Prolific IDs in the places they
    really appear, plus heavy material that must be excluded."""
    (exp_dir / "data").mkdir(parents=True)
    # participant_id is the anonymized index (keep); participant_id_str is the
    # raw Prolific worker id (must be dropped).
    (exp_dir / "data" / "responses.csv").write_text(
        "participant_id,participant_id_str,trial_index,sequence_a,chose_left\n"
        f"0,{FAKE_PID},0,HHTT,1\n"
        f"0,{FAKE_PID},1,THHT,0\n",
        encoding="utf-8",
    )
    (exp_dir / "data" / "observability.log").write_text(
        f"recruited participant PROLIFIC_PID={FAKE_PID} study={FAKE_STUDY_ID}\n",
        encoding="utf-8",
    )
    (exp_dir / "design").mkdir()
    (exp_dir / "design" / "design_rationale.md").write_text("why these stimuli", encoding="utf-8")
    (exp_dir / "deployment").mkdir()
    (exp_dir / "deployment" / "deployment_manifest.json").write_text(
        json.dumps({"prolific_study_id": FAKE_STUDY_ID,
                    "external_study_url": "https://x/?PROLIFIC_PID={{%PROLIFIC_PID%}}"}),
        encoding="utf-8",
    )
    (exp_dir / "cognitive_models").mkdir()
    (exp_dir / "cognitive_models" / "best.py").write_text("def p(): ...", encoding="utf-8")
    model_loop = exp_dir / "model_loop"
    model_loop.mkdir()
    (model_loop / "model_posterior.json").write_text(
        json.dumps(posterior if posterior is not None else _posterior_fixture()),
        encoding="utf-8",
    )
    (model_loop / "responses.csv").write_text(
        "participant_id,participant_id_str,h_a\n"
        f"0,{FAKE_PID},0.5\n",
        encoding="utf-8",
    )
    (model_loop / "iter_0" / "critique").mkdir(parents=True)
    (model_loop / "iter_0" / "critique" / "agent.jsonl").write_text(
        f'{{"msg": "saw {FAKE_PID}"}}\n', encoding="utf-8"
    )
    # Heavy material that must NOT be copied:
    fit_cache = model_loop / "iter_0" / "critique" / ".fit_cache"
    fit_cache.mkdir()
    (fit_cache / "trace.nc").write_bytes(b"\x00" * 4096)
    node_modules = model_loop / "node_modules" / "left-pad"
    node_modules.mkdir(parents=True)
    (node_modules / "index.js").write_text("junk", encoding="utf-8")


def _build_live_tree(root):
    """Two real runs (3 experiments each), a pilot, and a validation dir."""
    for run in ("run1", "run2"):
        for n in (1, 2, 3):
            _write_experiment(root / run / "data" / PROJECT / f"experiment{n}")
    _write_experiment(root / "pilot1" / "data" / PROJECT / "experiment1")
    _write_experiment(root / "_validate_sim" / "data" / PROJECT / "experiment1")
    (root / "runs" / "run1" / "repo" / "functions" / "node_modules").mkdir(parents=True)


# --- discovery --------------------------------------------------------------


def test_discover_runs_excludes_pilots_and_validation(tmp_path):
    _build_live_tree(tmp_path)
    assert discover_runs(tmp_path, project=PROJECT) == ["run1", "run2"]
    assert discover_runs(tmp_path, project=PROJECT, include_pilots=True) == [
        "pilot1",
        "run1",
        "run2",
    ]


# --- end-to-end collection --------------------------------------------------


def test_collect_copies_results_excluding_heavy_material(tmp_path):
    source = tmp_path / "scratch"
    source.mkdir()
    _build_live_tree(source)
    dest = tmp_path / "repo" / "data" / "results" / "human_experiment"

    report = collect_human_results(source, dest, project=PROJECT)

    assert (dest / "run1" / PROJECT / "experiment1" / "data" / "responses.csv").is_file()
    assert (dest / "run2" / PROJECT / "experiment3" / "model_loop" / "model_posterior.json").is_file()
    assert not (dest / "pilot1").exists()
    assert not (dest / "_validate_sim").exists()
    assert not list(dest.rglob("*.nc"))
    assert not list(dest.rglob(".fit_cache"))
    assert not list(dest.rglob("node_modules"))
    assert (dest / "SUMMARY.md").is_file()
    assert report.n_experiments == 2 * 3
    assert sorted(report.runs) == ["run1", "run2"]


def test_collect_scrubs_every_prolific_id(tmp_path):
    source = tmp_path / "scratch"
    source.mkdir()
    _build_live_tree(source)
    dest = tmp_path / "out"

    report = collect_human_results(source, dest, project=PROJECT)

    # No 24-hex Prolific id survives in ANY copied file.
    for path in dest.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert FAKE_PID not in text, path
            assert FAKE_STUDY_ID not in text, path
            assert not find_prolific_ids(text), path

    # The participant_id_str column is gone, but the anonymized index remains.
    resp = (dest / "run1" / PROJECT / "experiment1" / "data" / "responses.csv").read_text()
    assert "participant_id_str" not in resp
    assert "participant_id" in resp  # header still present
    assert report.n_ids_scrubbed > 0


# --- guardrails -------------------------------------------------------------


def test_collect_refuses_nonempty_destination_without_overwrite(tmp_path):
    source = tmp_path / "scratch"
    source.mkdir()
    _build_live_tree(source)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "stale").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        collect_human_results(source, dest, project=PROJECT)


def test_collect_fails_loudly_when_no_runs_found(tmp_path):
    source = tmp_path / "scratch"
    source.mkdir()
    with pytest.raises(FileNotFoundError, match="No .*runs"):
        collect_human_results(source, tmp_path / "out", project=PROJECT)


# --- pure redaction helpers -------------------------------------------------


def test_redact_text_replaces_ids_but_leaves_data_and_placeholders():
    text = f"pid={FAKE_PID} url={{{{%PROLIFIC_PID%}}}} seq=HHTTHHTT n=0.5123456789"
    out = redact_text(text)
    assert FAKE_PID not in out
    assert "PROLIFIC_PID" in out  # the templated placeholder is untouched
    assert "HHTTHHTT" in out  # H/T research data untouched
    assert "0.5123456789" in out


def test_redact_csv_text_drops_pii_column_and_redacts_residual_ids():
    csv_text = (
        "participant_id,participant_id_str,sequence_a\n"
        f"0,{FAKE_PID},HHTT\n"
    )
    out = redact_csv_text(csv_text)
    assert "participant_id_str" not in out
    assert FAKE_PID not in out
    assert "participant_id" in out and "sequence_a" in out
    assert "HHTT" in out


def test_find_prolific_ids_detects_and_redaction_clears_them():
    assert find_prolific_ids(f"a {FAKE_PID} b {FAKE_STUDY_ID}") == {FAKE_PID, FAKE_STUDY_ID}
    assert find_prolific_ids(redact_text(f"{FAKE_PID} {FAKE_STUDY_ID}")) == set()


# --- summary ----------------------------------------------------------------


def test_summarize_model_posterior_picks_winner_and_runner_up():
    s = summarize_model_posterior(_posterior_fixture())
    assert s["best_model"] == "iter1_candidate0"
    assert s["best_posterior"] == pytest.approx(0.999205)
    assert s["n_trials"] == 1280
    assert s["n_models"] == 3
    assert s["top_elpd_model"] == "iter1_candidate0"
    assert s["runner_up"] == "iter0_candidate0"
    assert s["runner_up_delta_elpd"] == pytest.approx(7.486, abs=1e-3)
    assert s["runner_up_dse"] == pytest.approx(4.011, abs=1e-3)


def test_render_summary_tabulates_runs_and_winners():
    records = [
        {"run": "run1", "experiment": "experiment1", "best_model": "iter1_candidate0",
         "best_posterior": 0.999, "n_trials": 1280, "n_responses": 1280,
         "n_participants": 32, "runner_up": "iter0_candidate0",
         "runner_up_delta_elpd": 7.49, "runner_up_dse": 4.01},
        {"run": "run2", "experiment": "experiment1", "best_model": "iter0_candidate1",
         "best_posterior": 0.8, "n_trials": 1280, "n_responses": 1280,
         "n_participants": 32, "runner_up": "prototype_similarity",
         "runner_up_delta_elpd": 3.1, "runner_up_dse": 2.0},
    ]
    md = render_human_experiment_summary(records, source="/scratch/x")
    assert "run1" in md and "run2" in md
    assert "iter1_candidate0" in md and "iter0_candidate1" in md
    assert "experiment1" in md
