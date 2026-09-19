"""Unit coverage for the recovery-improvement pieces the integration test
drives from the outside: the sweep digest's log/CSV parsing, the next_run.env
contract, campaign.env loading, and the real Slurm adapters' pure parts
(submit-output parsing, environment scrubbing, sbatch argv)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.recovery_improvement.campaign import Campaign, parse_env_file
from src.recovery_improvement.digest import (
    build_digest,
    parse_task_log,
    render_comparison,
    summarize_sweep,
)
from src.recovery_improvement.next_run import parse_next_run, sweep_env
from src.recovery_improvement.slurm import (
    parse_submit_output,
    review_sbatch_command,
    scrubbed_environment,
)

# --- digest -----------------------------------------------------------------


def test_task_log_done(tmp_path):
    log = tmp_path / "holdout_recovery_1_7.out"
    log.write_text(
        "[task 7] repeat=2 gt=motif_stack seed=102 -> /x\n"
        "  [warn] PSIS-LOO for 'a' is unreliable\n"
        "  [tokens] holdout recovery: 7,866,850 tokens over 24 LLM call(s)  cost=$7.67\n"
        "[task 7] done -> /x/holdout.json\n",
        encoding="utf-8",
    )
    outcome = parse_task_log(log)
    assert (outcome.task, outcome.repeat, outcome.gt) == (7, 2, "motif_stack")
    assert outcome.status == "done" and outcome.error == ""
    assert outcome.cost_usd == pytest.approx(7.67)


def test_task_log_failed_reports_last_error_line(tmp_path):
    log = tmp_path / "holdout_recovery_1_3.out"
    log.write_text(
        "[task 3] repeat=1 gt=falk_konold_dp seed=101 -> /x\n"
        "Traceback (most recent call last):\n"
        '  File "a.py", line 1\n'
        "ValueError: Stacking weights sum to 0.0\n",
        encoding="utf-8",
    )
    outcome = parse_task_log(log)
    assert outcome.status == "failed"
    assert outcome.error.startswith("ValueError: Stacking weights")


def test_task_log_incomplete_when_neither_done_nor_error(tmp_path):
    log = tmp_path / "holdout_recovery_1_4.out"
    log.write_text("[task 4] repeat=1 gt=motif_stack seed=101 -> /x\n  [inner-loop] fitting\n", encoding="utf-8")
    assert parse_task_log(log).status == "incomplete"


def test_missing_sweep_root_is_reported_not_skipped(tmp_path):
    summary = summarize_sweep(tmp_path / "nope", "iter1")
    assert summary.aggregate is None and summary.rows == []
    assert any("does not exist" in n for n in summary.notes)
    digest = build_digest([("iter1", tmp_path / "nope")], primary_label="iter1")
    assert "does not exist" in digest


def test_unfinished_sweep_without_aggregate_still_lists_rows(tmp_path):
    root = tmp_path / "sweep"
    cell = root / "run1" / "motif_stack"
    cell.mkdir(parents=True)
    (cell / "holdout.csv").write_text(
        "gt_model,experiment,step,iteration,global_step,best_model,pearson_r,rmse\n"
        "motif_stack,1,0,,0,seed,0.80,0.2\nmotif_stack,1,1,0,1,cand,0.90,0.1\n",
        encoding="utf-8",
    )
    summary = summarize_sweep(root)
    assert summary.aggregate is None
    assert any("analysis stage has not run" in n for n in summary.notes)
    (row,) = summary.rows
    assert (row.r_seed, row.r_final, row.best_final, row.n_steps) == (0.80, 0.90, "cand", 2)
    assert row.delta == pytest.approx(0.10)
    assert row.leakage == "no holdout.json"


def test_comparison_table_marks_missing_aggregates(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    (good / "test_retest.json").write_text(json.dumps({
        "gt_models": ["a"], "icc_2_1": 0.5,
        "per_gt_model": {"a": {"n_runs": 5, "mean": 0.9, "sd": 0.01}},
    }), encoding="utf-8")
    table = render_comparison([summarize_sweep(good, "base"), summarize_sweep(tmp_path / "none", "iter1")])
    assert "| a | 0.900 ± 0.010 (n=5) | n/a |" in table


def test_digest_run_row_accepts_legacy_csv_without_new_metric_columns(tmp_path):
    root = tmp_path / "sweep"
    cell = root / "run1" / "falk_konold_dp"
    cell.mkdir(parents=True)
    (cell / "holdout.csv").write_text(
        "gt_model,experiment,step,iteration,global_step,best_model,pearson_r,rmse,pearson_r_bma,rmse_bma\n"
        "falk_konold_dp,1,0,,0,seed,0.70,0.15,0.72,0.14\n",
        encoding="utf-8",
    )
    summary = summarize_sweep(root)
    (row,) = summary.rows
    assert row.rmse_final == pytest.approx(0.15)
    assert row.r_final == pytest.approx(0.70)


def test_build_digest_rejects_unknown_primary(tmp_path):
    with pytest.raises(ValueError, match="primary label"):
        build_digest([("x", tmp_path)], primary_label="y")


# --- next_run.env -------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "repo"
    cfg = repo / "scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("gt_models:\n  a:\n  b:\n", encoding="utf-8")
    return repo


DEFAULTS = {
    "N_REPEATS": "5", "BASE_SEED": "100", "GT_MODELS": "a b",
    "CONFIG": "scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml",
}


def test_next_run_empty_file_means_defaults(tmp_path, repo):
    path = tmp_path / "next_run.env"
    path.write_text("", encoding="utf-8")
    nr = parse_next_run(path, repo=repo, defaults=DEFAULTS)
    assert nr.values == {} and nr.note == ""


def test_next_run_note_is_kept_out_of_the_launch_env(tmp_path, repo):
    path = tmp_path / "next_run.env"
    path.write_text("NOTE=tests pruning\nDRAWS=1000\n", encoding="utf-8")
    nr = parse_next_run(path, repo=repo, defaults=DEFAULTS)
    assert nr.note == "tests pruning" and nr.values == {"DRAWS": "1000"}


def test_next_run_rejects_unknown_and_non_integer_keys(tmp_path, repo):
    path = tmp_path / "next_run.env"
    path.write_text("FOO=1\nDRAWS=many\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        parse_next_run(path, repo=repo, defaults=DEFAULTS)
    assert "unknown key 'FOO'" in str(exc.value) and "DRAWS must be an integer" in str(exc.value)


def test_next_run_rejects_gt_models_that_disagree_with_the_config(tmp_path, repo):
    path = tmp_path / "next_run.env"
    path.write_text("GT_MODELS=a c\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gt_models"):
        parse_next_run(path, repo=repo, defaults=DEFAULTS)


def test_next_run_rejects_missing_config(tmp_path, repo):
    path = tmp_path / "next_run.env"
    path.write_text("CONFIG=scripts/nope.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not exist"):
        parse_next_run(path, repo=repo, defaults=DEFAULTS)


# --- campaign.env -------------------------------------------------------------


def test_parse_env_file_handles_export_quotes_and_comments(tmp_path):
    path = tmp_path / "campaign.env"
    path.write_text('# c\nexport A="x y"\nB=\'z\'\nC=plain\n\n', encoding="utf-8")
    assert parse_env_file(path) == {"A": "x y", "B": "z", "C": "plain"}


def test_campaign_load_fails_loudly_on_missing_required_key(tmp_path):
    (tmp_path / "campaign.env").write_text("CAMPAIGN_NAME=x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        Campaign.load(tmp_path)


# --- slurm adapters (pure parts) -----------------------------------------------


def test_parse_submit_output_reads_the_three_job_ids():
    text = (
        "submitted setup job:    101\n"
        "submitted array job:    102 (1-20%5  =  5 repeats x 4 GTs)\n"
        "submitted analysis job: 103\n\nwatch with:  squeue --me\n"
    )
    jobs = parse_submit_output(text)
    assert (jobs.setup_id, jobs.array_id, jobs.analysis_id) == ("101", "102", "103")


def test_parse_submit_output_fails_loudly_when_a_stage_is_missing():
    with pytest.raises(RuntimeError, match="analysis"):
        parse_submit_output("submitted setup job: 1\nsubmitted array job: 2 (x)\n")


def test_scrubbed_environment_drops_slurm_and_sweep_state_but_keeps_the_rest():
    env = {
        "PATH": "/bin", "HOME": "/h", "SLURM_JOB_ID": "1", "SBATCH_PARTITION": "x",
        "UV_PROJECT_ENVIRONMENT": "/v", "VENV_PY": "/v/bin/python", "WORK_ROOT": "/w",
        "REPO": "/r", "N_REPEATS": "9", "ITERATION": "3", "XDG_DATA_HOME": "/x",
        "PYTENSOR_FLAGS": "f", "GOOGLE_API_KEY": "k",
    }
    scrubbed = scrubbed_environment(env)
    assert scrubbed == {"PATH": "/bin", "HOME": "/h", "GOOGLE_API_KEY": "k"}


def test_review_sbatch_command_chains_on_the_analysis_job(tmp_path):
    (tmp_path / "campaign.env").write_text(
        "CAMPAIGN_NAME=demo\nSOURCE_REPO=/src\nBASE_BRANCH=main\nBASELINE_ROOTS=/b\n"
        "MAX_ITERATIONS=3\nREVIEW_MODEL=m\nREVIEW_MAX_TURNS=1\nREVIEW_MAX_BUDGET_USD=1\n"
        "REVIEW_TIMEOUT_SEC=1\nMAX_REVIEW_REPAIRS=0\nREVIEW_PARTITION=hns\nREVIEW_TIME=04:00:00\n",
        encoding="utf-8",
    )
    campaign = Campaign.load(tmp_path)
    cmd = review_sbatch_command(campaign, 2, "555", "review")
    assert cmd[0] == "sbatch" and "--parsable" in cmd
    assert "--dependency=afterany:555" in cmd
    # The chain is unwatched, so a dying review job must announce itself.
    assert "--mail-type=FAIL,TIMEOUT" in cmd
    assert "--partition=hns" in cmd and "--time=04:00:00" in cmd
    export = next(a for a in cmd if a.startswith("--export="))
    assert f"CAMPAIGN_ROOT={tmp_path}" in export and "ITERATION=2" in export and "MODE=review" in export
    assert cmd[-1] == str(Path("/src/scripts/recovery_improvement/review_iteration.sbatch"))
    assert "--dependency" not in " ".join(review_sbatch_command(campaign, 1, None, "review"))


def test_sweep_env_merges_defaults_overrides_and_locations(tmp_path, repo):
    (tmp_path / "campaign.env").write_text(
        "CAMPAIGN_NAME=demo\nSOURCE_REPO=/src\nBASE_BRANCH=main\nBASELINE_ROOTS=/b\n"
        "MAX_ITERATIONS=3\nREVIEW_MODEL=m\nREVIEW_MAX_TURNS=1\nREVIEW_MAX_BUDGET_USD=1\n"
        "REVIEW_TIMEOUT_SEC=1\nMAX_REVIEW_REPAIRS=0\nSWEEP_N_REPEATS=5\nSWEEP_BASE_SEED=100\n",
        encoding="utf-8",
    )
    campaign = Campaign.load(tmp_path)
    path = tmp_path / "next_run.env"
    path.write_text("N_REPEATS=2\n", encoding="utf-8")
    nr = parse_next_run(path, repo=repo, defaults=campaign.sweep_defaults)
    env = sweep_env(nr, campaign, repo=repo, work_root=tmp_path / "sweep")
    assert env == {"N_REPEATS": "2", "BASE_SEED": "100", "REPO": str(repo), "WORK_ROOT": str(tmp_path / "sweep")}


# --- session limit -----------------------------------------------------------------

from datetime import datetime  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

from src.recovery_improvement.session_limit import (  # noqa: E402
    detect_session_limit,
    retry_begin_time,
)

LIMIT_MSG = "You've hit your session limit · resets 12am (America/Los_Angeles)"


def _begin_instant(begin: str) -> datetime:
    """The instant a ``retry_begin_time`` string names.

    ``retry_begin_time`` renders host-local wall-clock time on purpose: that is
    what ``sbatch --begin`` interprets. So its *text* differs by machine (Pacific
    on the cluster, UTC on a CI runner) while the instant it names does not.
    Compare instants, or the test only passes in one timezone.
    """
    return datetime.fromisoformat(begin).astimezone()


def test_session_limit_message_is_detected_with_its_reset_time():
    now = datetime(2026, 9, 7, 20, 2, tzinfo=ZoneInfo("America/Los_Angeles"))
    limit = detect_session_limit(LIMIT_MSG, now=now)
    assert limit is not None
    assert limit.reset_at == datetime(2026, 9, 8, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    assert _begin_instant(retry_begin_time(limit, now=now)) == datetime(
        2026, 9, 8, 0, 5, tzinfo=ZoneInfo("America/Los_Angeles")
    )


def test_session_limit_pm_time_already_past_rolls_to_tomorrow():
    now = datetime(2026, 9, 8, 16, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    limit = detect_session_limit("You've hit your usage limit · resets 3pm", now=now)
    assert limit.reset_at == datetime(2026, 9, 9, 15, 0, tzinfo=ZoneInfo("America/Los_Angeles"))


def test_session_limit_without_a_time_falls_back_to_an_hour():
    now = datetime(2026, 9, 8, 16, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    limit = detect_session_limit("You've reached your limit for now.", now=now)
    assert limit is not None and limit.reset_at is None
    assert _begin_instant(retry_begin_time(limit, now=now)) == datetime(
        2026, 9, 8, 17, 0, tzinfo=ZoneInfo("America/Los_Angeles")
    )


def test_ordinary_result_text_is_not_a_limit():
    assert detect_session_limit("Done. Wrote prescription.md and next_run.env.") is None


def test_other_limit_phrasings_and_relative_reset_times_are_parsed():
    now = datetime(2026, 9, 8, 16, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    limit = detect_session_limit("Usage limit exceeded. Try again in 4h 30m.", now=now)
    assert limit is not None
    assert limit.reset_at == datetime(2026, 9, 8, 20, 30, tzinfo=ZoneInfo("America/Los_Angeles"))
    limit = detect_session_limit("You have reached your weekly limit; available again in 1 hour and 5 minutes", now=now)
    assert limit.reset_at == datetime(2026, 9, 8, 17, 5, tzinfo=ZoneInfo("America/Los_Angeles"))
    limit = detect_session_limit("Rate limit reached, try again at 9 pm", now=now)
    assert limit.reset_at == datetime(2026, 9, 8, 21, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    # A note that merely discusses limits, past the first 200 characters, is not a hit.
    assert detect_session_limit("x" * 250 + " we hit your usage limit of candidates per round") is None


def test_an_unusable_model_is_detected_and_is_not_a_session_limit():
    """Claude Code exits 0 with this as its result text when the configured
    model is not available to the login (which happened when a re-login
    dropped access to claude-fable-5-1 mid-campaign). No repair round can fix
    it, so it must be told apart from a session limit and from a genuine
    missing deliverable."""
    from src.recovery_improvement.session_limit import (
        detect_session_limit,
        detect_unusable_model,
    )

    msg = (
        "There's an issue with the selected model (claude-fable-5-1). It may not "
        "exist or you may not have access to it. Run --model to pick a different model."
    )
    assert detect_unusable_model(msg) == "claude-fable-5-1"
    assert detect_session_limit(msg) is None
    assert detect_unusable_model("Done. Wrote prescription.md.") is None
    # A note that merely discusses model access is not a configuration failure.
    assert detect_unusable_model("x" * 250 + " issue with the selected model (foo)") is None
