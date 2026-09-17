"""Unit coverage for the consolidation driver's pure parts: the phase table,
progress markers, done-marker validation, the jobs files a phase hands to the
driver, prompt composition, the requeue argv, walltime parsing and the squeue
parser.

Everything that spawns Claude or talks to Slurm lives in
``scripts/consolidation/run_consolidation.py`` and is not exercised here."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.consolidation.driver import (
    BASE_DISALLOWED_TOOLS,
    PHASES,
    Blocked,
    compose_prompt,
    disallowed_tools,
    jobs_file_owner,
    jobs_still_queued,
    load_env,
    needs_walltime_requeue,
    newly_reopened,
    next_phase,
    parse_jobs_file,
    phase_by_id,
    phase_round,
    phase_state,
    requeue_command,
    retry_markers,
    seconds_left,
    validate_done_marker,
)

SHA = "0123456789abcdef0123456789abcdef01234567"


# --- phase table --------------------------------------------------------------


def test_phases_run_p0_to_p21_in_order_and_only_submitting_phases_may_sbatch():
    assert [p.id for p in PHASES] == [f"P{k}" for k in range(22)]
    assert [p.id for p in PHASES if p.allows_sbatch] == ["P7", "P12", "P14", "P15", "P20"]
    assert phase_by_id("P3").title
    assert phase_by_id("P11").title.startswith("Simplify")
    with pytest.raises(KeyError):
        phase_by_id("P22")


def test_each_waiting_phase_waits_for_a_jobs_file_an_earlier_phase_writes():
    seen: set[str] = set()
    for phase in PHASES:
        if phase.waits_for:
            owner = jobs_file_owner(phase.waits_for)
            assert owner.jobs_file == phase.waits_for
            assert PHASES.index(owner) < PHASES.index(phase)
        if phase.jobs_file:
            assert phase.jobs_file not in seen, "two phases write the same jobs file"
            seen.add(phase.jobs_file)
            assert phase.allows_sbatch and phase.required_labels
    assert phase_by_id("P7").jobs_file == "smoke_jobs.json"
    assert phase_by_id("P8").waits_for == "smoke_jobs.json"
    assert phase_by_id("P12").jobs_file == "isolation_smoke_jobs.json"
    assert phase_by_id("P13").waits_for == "isolation_smoke_jobs.json"
    assert phase_by_id("P14").required_labels == ("raw",)
    assert phase_by_id("P15").waits_for == "sweep_jobs.json"
    assert phase_by_id("P16").waits_for == "analysis_jobs.json"
    assert phase_by_id("P20").jobs_file == "cleanup_smoke_jobs.json"
    assert phase_by_id("P21").waits_for == "cleanup_smoke_jobs.json"
    assert phase_by_id("P21").requires_files == ("CLEANUP_REPORT.md",)
    with pytest.raises(KeyError):
        jobs_file_owner("nobody_writes_this.json")


def test_disallowed_tools_add_sbatch_except_in_submitting_phases():
    assert "Bash(sbatch:*)" in disallowed_tools(phase_by_id("P2"))
    assert "Bash(sbatch:*)" in disallowed_tools(phase_by_id("P8"))
    for phase_id in ("P7", "P12", "P14", "P15", "P20"):
        assert "Bash(sbatch:*)" not in disallowed_tools(phase_by_id(phase_id))
        for tool in BASE_DISALLOWED_TOOLS:
            assert tool in disallowed_tools(phase_by_id(phase_id))


# --- progress markers ---------------------------------------------------------


def test_phase_state_reads_done_and_blocked_markers(tmp_path):
    progress = tmp_path / "progress"
    progress.mkdir()
    assert phase_state(progress, "P0") == "pending"
    (progress / "P0.done").write_text(f"commit: {SHA}\n")
    assert phase_state(progress, "P0") == "done"
    (progress / "P1.blocked").write_text("cannot reconcile\n")
    assert phase_state(progress, "P1") == "blocked"
    # blocked wins over done: a phase re-opened by a later stop condition stays blocked
    (progress / "P1.done").write_text(f"commit: {SHA}\n")
    assert phase_state(progress, "P1") == "blocked"


def test_next_phase_is_the_first_undone_and_a_blocked_one_raises(tmp_path):
    progress = tmp_path / "progress"
    progress.mkdir()
    assert next_phase(progress).id == "P0"
    (progress / "P0.done").write_text(f"commit: {SHA}\n")
    (progress / "P1.done").write_text(f"commit: {SHA}\n")
    assert next_phase(progress).id == "P2"
    (progress / "P2.blocked").write_text("stop condition\n")
    with pytest.raises(Blocked) as exc:
        next_phase(progress)
    assert "P2" in str(exc.value) and "stop condition" in str(exc.value)


def test_next_phase_is_none_when_everything_is_done(tmp_path):
    progress = tmp_path / "progress"
    progress.mkdir()
    for phase in PHASES:
        (progress / f"{phase.id}.done").write_text(f"commit: {SHA}\n")
    assert next_phase(progress) is None


def test_phase_round_counts_retry_markers(tmp_path):
    progress = tmp_path / "progress"
    progress.mkdir()
    assert phase_round(progress, "P7") == 1
    (progress / "P7.retry").write_text("verifier path bug\n")
    assert phase_round(progress, "P7") == 2
    (progress / "P7.retry1").write_text("second\n")
    assert phase_round(progress, "P7") == 3
    assert phase_round(progress, "P9") == 1
    assert phase_by_id("P7").max_rounds == 2
    assert phase_by_id("P12").max_rounds == 3
    assert phase_by_id("P9").max_rounds is None


def test_a_verdict_phase_can_reopen_an_earlier_phase(tmp_path):
    """P8 writes P7.retry and deletes P7.done: the driver must treat P8's
    session as finished WITHOUT demanding P8.done, and run P7 again."""
    progress = tmp_path / "progress"
    progress.mkdir()
    before = retry_markers(progress)
    assert before == set()
    (progress / "P7.retry").write_text("manifest scrub bug\n")
    after = retry_markers(progress)
    assert after == {"P7.retry"}
    assert newly_reopened("P8", before, after) == "P7"
    # a marker for a LATER phase, or no new marker, re-opens nothing
    assert newly_reopened("P8", after, after) is None
    (progress / "P11.retry").write_text("x\n")
    assert newly_reopened("P8", after, retry_markers(progress)) is None
    # the earliest re-opened phase wins when several appear
    (progress / "P2.retry").write_text("y\n")
    assert newly_reopened("P13", set(), retry_markers(progress)) == "P2"


# --- done-marker validation ---------------------------------------------------


def _write_done(progress: Path, phase_id: str, sha: str = SHA) -> None:
    (progress / f"{phase_id}.done").write_text(f"commit: {sha}\nsummary line\n")


def _jobs(labels: dict[str, list[str]]) -> str:
    return json.dumps({
        label: {"work_root": f"/w/{label}", "job_ids": ids} for label, ids in labels.items()
    })


def test_done_marker_must_exist_name_head_and_have_a_clean_tree(tmp_path):
    progress = tmp_path / "progress"
    progress.mkdir()
    problems = validate_done_marker(progress, "P2", head_sha=SHA, porcelain="", work_root=tmp_path)
    assert any("P2.done" in p and "missing" in p for p in problems)

    _write_done(progress, "P2", sha="deadbeef")
    problems = validate_done_marker(progress, "P2", head_sha=SHA, porcelain="", work_root=tmp_path)
    assert any("deadbeef" in p and SHA in p for p in problems)

    _write_done(progress, "P2")
    problems = validate_done_marker(
        progress, "P2", head_sha=SHA, porcelain=" M src/x.py\n?? notes.txt", work_root=tmp_path
    )
    assert any("uncommitted" in p and "src/x.py" in p for p in problems)

    assert validate_done_marker(progress, "P2", head_sha=SHA, porcelain="", work_root=tmp_path) == []


def test_done_marker_first_line_must_be_the_commit_line(tmp_path):
    progress = tmp_path / "progress"
    progress.mkdir()
    (progress / "P4.done").write_text("all good\n")
    problems = validate_done_marker(progress, "P4", head_sha=SHA, porcelain="", work_root=tmp_path)
    assert any("commit:" in p for p in problems)


def test_p0_requires_the_baseline_files(tmp_path):
    progress = tmp_path / "progress"
    progress.mkdir()
    _write_done(progress, "P0")
    problems = validate_done_marker(progress, "P0", head_sha=SHA, porcelain="", work_root=tmp_path)
    assert any("baseline_failing_tests.txt" in p for p in problems)
    (progress / "baseline_failing_tests.txt").write_text("tests/test_a.py::test_x\n")
    (progress / "baseline_collection_errors.txt").write_text("")
    assert validate_done_marker(progress, "P0", head_sha=SHA, porcelain="", work_root=tmp_path) == []


def test_a_submitting_phase_requires_its_jobs_file_with_every_label(tmp_path):
    progress = tmp_path / "progress"
    progress.mkdir()
    _write_done(progress, "P7")
    problems = validate_done_marker(progress, "P7", head_sha=SHA, porcelain="", work_root=tmp_path)
    assert any("smoke_jobs.json" in p for p in problems)
    (progress / "smoke_jobs.json").write_text(_jobs({"featurized": ["1", "2", "3"]}))
    problems = validate_done_marker(progress, "P7", head_sha=SHA, porcelain="", work_root=tmp_path)
    assert any("raw" in p for p in problems)
    (progress / "smoke_jobs.json").write_text(
        _jobs({"featurized": ["1", "2", "3"], "raw": ["4", "5", "6", "7"]})
    )
    assert validate_done_marker(progress, "P7", head_sha=SHA, porcelain="", work_root=tmp_path) == []

    _write_done(progress, "P12")
    (progress / "isolation_smoke_jobs.json").write_text(_jobs({"raw": ["8", "9", "10", "11"]}))
    assert validate_done_marker(progress, "P12", head_sha=SHA, porcelain="", work_root=tmp_path) == []
    _write_done(progress, "P14")
    (progress / "sweep_jobs.json").write_text(_jobs({"raw": ["8", "9", "10", "11"]}))
    assert validate_done_marker(progress, "P14", head_sha=SHA, porcelain="", work_root=tmp_path) == []
    _write_done(progress, "P15")
    (progress / "analysis_jobs.json").write_text(_jobs({"analysis": ["12"]}))
    assert validate_done_marker(progress, "P15", head_sha=SHA, porcelain="", work_root=tmp_path) == []


def test_verdict_and_results_phases_require_their_report_files(tmp_path):
    progress = tmp_path / "progress"
    progress.mkdir()
    _write_done(progress, "P8")
    problems = validate_done_marker(progress, "P8", head_sha=SHA, porcelain="", work_root=tmp_path)
    assert any("VERDICT.md" in p for p in problems) and any("HANDOFF.md" in p for p in problems)
    (tmp_path / "VERDICT.md").write_text("pass\n")
    (tmp_path / "HANDOFF.md").write_text("fetch it\n")
    assert validate_done_marker(progress, "P8", head_sha=SHA, porcelain="", work_root=tmp_path) == []

    _write_done(progress, "P13")
    assert validate_done_marker(progress, "P13", head_sha=SHA, porcelain="", work_root=tmp_path) == []

    _write_done(progress, "P16")
    problems = validate_done_marker(progress, "P16", head_sha=SHA, porcelain="", work_root=tmp_path)
    assert any("RESULTS.md" in p for p in problems)
    (tmp_path / "RESULTS.md").write_text("rmse table\n")
    assert validate_done_marker(progress, "P16", head_sha=SHA, porcelain="", work_root=tmp_path) == []


# --- jobs files -------------------------------------------------------------------


def test_parse_jobs_file_returns_every_id_and_rejects_bad_input(tmp_path):
    path = tmp_path / "smoke_jobs.json"
    path.write_text(_jobs({"featurized": ["11", "12", "13"], "raw": ["21", "22", "23", "24"]}))
    assert parse_jobs_file(path, ("featurized", "raw")) == ["11", "12", "13", "21", "22", "23", "24"]
    path.write_text(_jobs({"featurized": ["abc"], "raw": ["1"]}))
    with pytest.raises(ValueError, match="non-numeric"):
        parse_jobs_file(path, ("featurized", "raw"))
    path.write_text(_jobs({"featurized": ["1"]}))
    with pytest.raises(ValueError, match="raw"):
        parse_jobs_file(path, ("featurized", "raw"))
    with pytest.raises(ValueError, match="missing"):
        parse_jobs_file(tmp_path / "absent.json", ("raw",))
    path.write_text("{not json")
    with pytest.raises(ValueError, match="JSON"):
        parse_jobs_file(path, ("raw",))


def test_jobs_still_queued_collapses_array_task_ids():
    out = "12345_1\n12345_[2-4%2]\n67890\n"
    assert jobs_still_queued(out) == {"12345", "67890"}
    assert jobs_still_queued("") == set()


# --- prompt, env, requeue -------------------------------------------------------


def test_compose_prompt_substitutes_and_rejects_unknown_placeholders():
    text = compose_prompt("phase $phase_id in $repo", {"phase_id": "P1", "repo": "/r"})
    assert text == "phase P1 in /r"
    with pytest.raises(ValueError, match="missing_key"):
        compose_prompt("$missing_key", {})


def test_load_env_reads_key_values_and_ignores_comments(tmp_path):
    path = tmp_path / "consolidation.env"
    path.write_text('# header\nA=1\nB="two words"\nC=x=y\n\n')
    assert load_env(path) == {"A": "1", "B": "two words", "C": "x=y"}
    with pytest.raises(ValueError):
        load_env(tmp_path / "missing.env")


def test_requeue_command_carries_dependency_and_begin():
    cmd = requeue_command(
        Path("/s/consolidate.sbatch"),
        job_name="consolidate_2026_09",
        partition="normal",
        time="2-00:00:00",
        cpus="4",
        mem="16GB",
        log_dir=Path("/w/slurm_logs"),
        export="ALL,WORK_ROOT=/w",
        dependency="afterany:1:2",
        begin="2026-09-17T00:05:00",
    )
    assert cmd[0:2] == ["sbatch", "--parsable"]
    assert "--dependency=afterany:1:2" in cmd
    assert "--begin=2026-09-17T00:05:00" in cmd
    assert "--export=ALL,WORK_ROOT=/w" in cmd
    assert cmd[-1] == "/s/consolidate.sbatch"
    plain = requeue_command(
        Path("/s/x.sbatch"), job_name="j", partition="p", time="t", cpus="1", mem="1G",
        log_dir=Path("/l"), export="ALL",
    )
    assert not any(a.startswith("--dependency") or a.startswith("--begin") for a in plain)


# --- walltime ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [("1-02:03:04", 93784), ("02:03:04", 7384), ("03:04", 184), ("7", 7)],
)
def test_seconds_left_parses_squeue_time_formats(text, expected):
    assert seconds_left(text) == expected


def test_seconds_left_returns_none_for_unlimited_and_raises_on_garbage():
    assert seconds_left("UNLIMITED") is None
    assert seconds_left("NOT_SET") is None
    with pytest.raises(ValueError):
        seconds_left("soon")


def test_needs_walltime_requeue_when_a_session_would_not_fit():
    assert needs_walltime_requeue(seconds_left=3600, session_timeout=21600, margin=900)
    assert not needs_walltime_requeue(seconds_left=30000, session_timeout=21600, margin=900)
    assert not needs_walltime_requeue(seconds_left=None, session_timeout=21600, margin=900)
