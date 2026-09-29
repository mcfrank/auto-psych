"""Retries of a holdout sweep resume on the sweep's code, with the resources
they were meant to have.

A retry used to resubmit the setup job, which re-staged ``harness_repo`` from
the live checkout days after the cells started; its 128 GB request for the
out-of-memory group was set as ``SBATCH_MEM_PER_NODE``, which sbatch reads
from the environment, so it reached the setup, retry and analysis jobs and
(through ``--export=ALL``) every later round's same-memory tasks; and the
retry array dropped the ``%MAX_PARALLEL`` cap. These tests run the submit and
retry scripts against fake ``sbatch`` / ``sacct`` commands that record what
they were asked to do.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.paths import REPO_ROOT

SLURM_DIR = REPO_ROOT / "scripts" / "subjective_randomness" / "slurm"

# Records each call's arguments and the memory variables it inherited; prints
# a job id.
FAKE_SBATCH = """#!/bin/bash
n=$(( $(ls "$FAKE_LOG_DIR" | wc -l) + 1 ))
{ printf '%s\\n' "$@"; echo "ENV SBATCH_MEM_PER_NODE=${SBATCH_MEM_PER_NODE:-}";
  echo "ENV ARRAY_MEM=${ARRAY_MEM:-}"; } > "$FAKE_LOG_DIR/$n"
echo "$(( 1000 + n ))"
"""


def _bin_dir(tmp_path: Path, **scripts: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in scripts.items():
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    return bin_dir


def _submissions(log_dir: Path) -> list[dict]:
    """Each sbatch call: its script (last argument), arguments and inherited
    memory variables."""
    calls = []
    for path in sorted(log_dir.iterdir(), key=lambda p: int(p.name)):
        lines = path.read_text(encoding="utf-8").splitlines()
        args = [line for line in lines if not line.startswith("ENV ")]
        env = dict(line[4:].split("=", 1) for line in lines if line.startswith("ENV "))
        calls.append({"script": args[-1], "args": args[:-1], "env": env})
    return calls


def _run_submit(
    tmp_path: Path, script: str = "submit_holdout_test_retest.sh", **env_overrides: str
) -> subprocess.CompletedProcess:
    log_dir = tmp_path / "sbatch_calls"
    log_dir.mkdir(exist_ok=True)
    bin_dir = _bin_dir(tmp_path, sbatch=FAKE_SBATCH)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "WORK_ROOT": str(tmp_path / "work"),
        "FAKE_LOG_DIR": str(log_dir),
        **env_overrides,
    }
    return subprocess.run(
        ["bash", str(SLURM_DIR / script)],
        env=env, capture_output=True, text=True, timeout=30,
    )


def _by_script(calls: list[dict], script: str) -> list[dict]:
    return [c for c in calls if c["script"] == script]


def test_a_first_submission_stages_the_code_and_caps_the_array(tmp_path):
    result = _run_submit(tmp_path)
    assert result.returncode == 0, result.stderr
    calls = _submissions(tmp_path / "sbatch_calls")
    assert len(_by_script(calls, "holdout_setup.sbatch")) == 1
    (array,) = _by_script(calls, "holdout_recovery_array.sbatch")
    assert "--array=1-20%5" in array["args"]
    assert "--dependency=afterok:1001" in array["args"]
    assert not any(a.startswith("--mem") for a in array["args"])


def test_a_retry_skips_setup_keeps_the_cap_and_asks_for_memory_only_for_its_array(tmp_path):
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "code_commit").write_text("abc123\n", encoding="utf-8")
    result = _run_submit(
        tmp_path, RETRY_ROUND="1", ARRAY_TASKS="3,7", ARRAY_MEM="128G", MAX_PARALLEL="4"
    )
    assert result.returncode == 0, result.stderr
    calls = _submissions(tmp_path / "sbatch_calls")
    # No setup job: it would re-stage (or refuse) the live checkout.
    assert _by_script(calls, "holdout_setup.sbatch") == []
    (array,) = _by_script(calls, "holdout_recovery_array.sbatch")
    assert "--array=3,7%4" in array["args"]
    assert "--mem=128G" in array["args"]
    assert not any(a.startswith("--dependency") for a in array["args"])
    # The memory request reaches no other job, neither as a flag nor through
    # the environment sbatch (and --export=ALL) would pass on.
    for call in calls:
        assert call["env"] == {"SBATCH_MEM_PER_NODE": "", "ARRAY_MEM": ""}, call
        if call is not array:
            assert not any(a.startswith("--mem") for a in call["args"]), call


def test_a_retry_without_staged_code_fails_loudly(tmp_path):
    result = _run_submit(tmp_path, RETRY_ROUND="1", ARRAY_TASKS="3")
    assert result.returncode != 0
    assert "code_commit" in result.stderr
    assert _submissions(tmp_path / "sbatch_calls") == []


# ── holdout_retry.sbatch ─────────────────────────────────────────────


def _staged_harness(tmp_path: Path) -> Path:
    """harness_repo with the retry's collaborators: a stub _env.sh, the real
    cell_status.py, and a submit script that records how it was called."""
    staged = tmp_path / "work" / "harness_repo" / "scripts" / "subjective_randomness" / "slurm"
    staged.mkdir(parents=True)
    (staged / "_env.sh").write_text(f'export VENV_PY="{sys.executable}"\n', encoding="utf-8")
    shutil.copy(SLURM_DIR / "cell_status.py", staged / "cell_status.py")
    (staged / "submit_holdout_test_retest.sh").write_text(
        'echo "tasks=$ARRAY_TASKS mem=${ARRAY_MEM:-} round=$RETRY_ROUND" >> "$SUBMIT_LOG"\n',
        encoding="utf-8",
    )
    return staged


def test_the_retry_resumes_on_the_staged_scripts_with_memory_per_group(tmp_path):
    _staged_harness(tmp_path)
    work = tmp_path / "work"
    # Task 2 (run1/b) finished although Slurm marked it OUT_OF_MEMORY.
    (work / "run1" / "b").mkdir(parents=True)
    (work / "run1" / "b" / "holdout.json").write_text(json.dumps({}), encoding="utf-8")
    sacct = "#!/bin/bash\nprintf '9_1|FAILED\\n9_2|OUT_OF_MEMORY\\n9_3|TIMEOUT\\n9_4|OUT_OF_MEMORY\\n'\n"
    bin_dir = _bin_dir(tmp_path, sacct=sacct)
    submit_log = tmp_path / "submit.log"
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "WORK_ROOT": str(work),
        "GT_MODELS": "a b",
        "RETRY_ARRAY_ID": "9",
        "RETRY_ROUND": "1",
        # An earlier round's out-of-memory value, inherited through --export=ALL.
        "ARRAY_MEM": "128G",
        "SUBMIT_LOG": str(submit_log),
        "RETRY_SUBMIT_SCRIPT": "submit_holdout_test_retest.sh",
        # The live checkout's scripts must not be what runs.
        "HOLDOUT_SLURM_DIR": str(tmp_path / "live_checkout_that_does_not_exist"),
    }
    result = subprocess.run(
        ["bash", str(SLURM_DIR / "holdout_retry.sbatch")],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert submit_log.read_text(encoding="utf-8").splitlines() == [
        "tasks=1,3 mem= round=2",
        "tasks=4 mem=128G round=2",
    ]


def test_the_retry_plan_skips_a_task_whose_cell_finished(tmp_path):
    from scripts.subjective_randomness.slurm.cell_status import retry_plan

    (tmp_path / "run2" / "a").mkdir(parents=True)
    (tmp_path / "run2" / "a" / "holdout.json").write_text("{}", encoding="utf-8")
    plan = retry_plan(
        "5_3|TIMEOUT\n5_4|OUT_OF_MEMORY\n", work_root=tmp_path, gt_models=["a", "b"]
    )
    assert plan == {"same_memory": [], "more_memory": [4]}


@pytest.mark.parametrize("script", ["holdout_recovery_array.sbatch", "holdout_retry.sbatch"])
def test_jobs_after_setup_read_no_code_from_the_live_checkout(script):
    """They source the staged scripts; the array builds agent trees from the
    staged agent source."""
    text = (SLURM_DIR / script).read_text(encoding="utf-8")
    assert 'SLURM_DIR="$HARNESS_REPO/scripts/subjective_randomness/slurm"' in text
    assert "HOLDOUT_SLURM_DIR" not in text
    assert '"$REPO"/' not in text
    assert '"$REPO/scripts' not in text


# ── holdout_recovery_array.sbatch: a cell's code and a finished cell ──


def _run_array_task(tmp_path: Path, *, staged_code: str = "abc123") -> subprocess.CompletedProcess:
    """Task 1 (run1/gtone) of the array against a staged harness whose _env.sh is
    a stub. It stops, at the latest, at the missing GT snapshot."""
    staged = _staged_harness(tmp_path)
    shutil.copy(SLURM_DIR / "cell_lock.sh", staged / "cell_lock.sh")
    work = tmp_path / "work"
    (work / "agent_src").mkdir()
    (work / "code_commit").write_text(f"{staged_code}\n", encoding="utf-8")
    bin_dir = _bin_dir(tmp_path, squeue="#!/bin/bash\n")
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "WORK_ROOT": str(work),
        "GT_MODELS": "gtone gttwo",
        "SLURM_ARRAY_TASK_ID": "1",
        "SLURM_JOB_ID": "77",
        "AGENT_TREES_ROOT": str(tmp_path / "agent_trees"),
    }
    return subprocess.run(
        ["bash", str(SLURM_DIR / "holdout_recovery_array.sbatch")],
        env=env, capture_output=True, text=True, timeout=60,
    )


def test_a_new_cell_records_the_code_it_starts_on(tmp_path):
    result = _run_array_task(tmp_path)
    assert "GT snapshot missing" in result.stderr  # got past the code check
    assert (tmp_path / "work" / "run1" / "gtone" / "code_commit").read_text().strip() == "abc123"


def test_a_cell_is_not_resumed_on_other_code(tmp_path):
    cell = tmp_path / "work" / "run1" / "gtone"
    cell.mkdir(parents=True)
    (cell / "code_commit").write_text("old999\n", encoding="utf-8")
    result = _run_array_task(tmp_path)
    assert result.returncode != 0
    assert "refusing to resume it on other code" in result.stderr


def test_a_cell_with_earlier_work_and_no_code_record_is_not_resumed(tmp_path):
    cell = tmp_path / "work" / "run1" / "gtone"
    cell.mkdir(parents=True)
    (cell / "mcmc_cache").mkdir()
    result = _run_array_task(tmp_path)
    assert result.returncode != 0
    assert "no record of the code" in result.stderr


def test_a_finished_cell_is_left_untouched(tmp_path):
    """A retry of a finished cell used to rebuild an empty _runs/ and overwrite
    the real agent_runs.tar.gz with it."""
    cell = tmp_path / "work" / "run1" / "gtone"
    cell.mkdir(parents=True)
    (cell / "holdout.json").write_text("{}", encoding="utf-8")
    (cell / "agent_runs.tar.gz").write_bytes(b"the real archive")
    (cell / "gt_name_mentions.txt").write_text("x.py\n", encoding="utf-8")
    result = _run_array_task(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "already complete" in result.stdout
    assert (cell / "agent_runs.tar.gz").read_bytes() == b"the real archive"
    assert (cell / "gt_name_mentions.txt").read_text() == "x.py\n"
    assert not (cell / "repo").exists()


def test_the_agent_tree_has_a_random_id_recorded_in_the_cell(tmp_path):
    """The id used to be a hash of the GT-named cell path, which an agent
    that found WORK_ROOT could recompute for each candidate ground truth."""
    import hashlib
    import re

    _run_array_task(tmp_path)
    cell = tmp_path / "work" / "run1" / "gtone"
    tree_id = (cell / "agent_tree_id").read_text(encoding="utf-8")
    assert re.fullmatch(r"[0-9a-f]{16}", tree_id)
    assert tree_id != hashlib.sha256(str(cell).encode()).hexdigest()[:16]
    assert (cell / "repo").resolve() == (tmp_path / "agent_trees" / tree_id / "repo").resolve()

    # A resubmitted task finds the same tree.
    (tmp_path / "work" / "code_commit").unlink()
    shutil.rmtree(tmp_path / "work" / "harness_repo")
    shutil.rmtree(tmp_path / "work" / "agent_src")
    _run_array_task(tmp_path)
    assert (cell / "agent_tree_id").read_text(encoding="utf-8") == tree_id


# ── stage_sweep_code.sh: the sweep's code, staged once ────────────────


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo, check=True, capture_output=True,
    )


def _checkout(tmp_path: Path) -> Path:
    """A git checkout carrying the staging scripts, a src/ file and a test."""
    repo = tmp_path / "checkout"
    slurm = repo / "scripts" / "subjective_randomness" / "slurm"
    slurm.mkdir(parents=True)
    for name in ("code_commit.sh", "agent_tree.exclude"):
        shutil.copy(SLURM_DIR / name, slurm / name)
    (repo / "src").mkdir()
    (repo / "src" / "loop.py").write_text("VERSION = 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_loop.py").write_text("# names the ground truth\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first")
    return repo


def _stage(repo: Path, work: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SLURM_DIR / "stage_sweep_code.sh")],
        env={**os.environ, "REPO": str(repo), "WORK_ROOT": str(work)},
        capture_output=True, text=True, timeout=60,
    )


def test_the_code_is_staged_once_and_a_changed_checkout_is_refused(tmp_path):
    repo, work = _checkout(tmp_path), tmp_path / "work"

    first = _stage(repo, work)
    assert first.returncode == 0, first.stderr
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()
    assert (work / "code_commit").read_text().strip() == head
    assert (work / "harness_repo" / "tests" / "test_loop.py").exists()
    assert (work / "harness_repo" / ".here").exists()
    assert (work / "agent_src" / "src" / "loop.py").exists()
    assert not (work / "agent_src" / "tests").exists()  # agent_tree.exclude

    again = _stage(repo, work)
    assert again.returncode == 0 and "not re-staging" in again.stdout

    (repo / "src" / "loop.py").write_text("VERSION = 2\n", encoding="utf-8")
    dirty = _stage(repo, work)
    assert dirty.returncode != 0 and "staged from code" in dirty.stderr
    assert (work / "harness_repo" / "src" / "loop.py").read_text() == "VERSION = 1\n"


def test_cells_without_a_code_record_are_not_restaged_under(tmp_path):
    repo, work = _checkout(tmp_path), tmp_path / "work"
    (work / "run1" / "motif_stack").mkdir(parents=True)
    result = _stage(repo, work)
    assert result.returncode != 0 and "no record of the code" in result.stderr
    assert not (work / "harness_repo").exists()


# ── The impossible-model sweep: the same retry and summary machinery ──
#
# Its submitter used to submit setup, array and analysis only, so a cell that
# timed out or ran out of memory needed a manual resubmission (AUDIT_STATUS,
# "New since the audits" 2).

IMPOSSIBLE = "submit_impossible_holdout_test_retest.sh"


def test_the_impossible_sweep_chains_the_shared_retry_job_and_a_summary(tmp_path):
    result = _run_submit(tmp_path, IMPOSSIBLE)
    assert result.returncode == 0, result.stderr
    calls = _submissions(tmp_path / "sbatch_calls")
    assert [c["script"] for c in calls] == [
        "impossible_holdout_setup.sbatch",
        "impossible_holdout_recovery_array.sbatch",
        "holdout_retry.sbatch",
        "holdout_analysis.sbatch",
    ]
    array, retry = calls[1], calls[2]
    assert "--array=1-20%5" in array["args"]
    assert "--dependency=afterany:1002" in retry["args"]
    assert any("RETRY_ARRAY_ID=1002" in a and "RETRY_ROUND=0" in a for a in retry["args"])
    assert "--dependency=afterany:1002" in calls[3]["args"]


def test_an_impossible_retry_skips_setup_keeps_the_cap_and_the_memory_to_its_array(tmp_path):
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "code_commit").write_text("abc123\n", encoding="utf-8")
    result = _run_submit(
        tmp_path, IMPOSSIBLE, RETRY_ROUND="2", ARRAY_TASKS="3,7", ARRAY_MEM="128G",
        MAX_PARALLEL="4",
    )
    assert result.returncode == 0, result.stderr
    calls = _submissions(tmp_path / "sbatch_calls")
    assert _by_script(calls, "impossible_holdout_setup.sbatch") == []
    # The last round (2 of 2) chains no further retry.
    assert _by_script(calls, "holdout_retry.sbatch") == []
    (array,) = _by_script(calls, "impossible_holdout_recovery_array.sbatch")
    assert "--array=3,7%4" in array["args"] and "--mem=128G" in array["args"]
    for call in calls:
        assert call["env"] == {"SBATCH_MEM_PER_NODE": "", "ARRAY_MEM": ""}, call
        if call is not array:
            assert not any(a.startswith("--mem") for a in call["args"]), call


def test_the_old_mem_knob_fails_loudly(tmp_path):
    result = _run_submit(tmp_path, IMPOSSIBLE, MEM="64GB")
    assert result.returncode != 0 and "ARRAY_MEM" in result.stderr
    assert _submissions(tmp_path / "sbatch_calls") == []


def test_the_retry_job_resubmits_through_the_impossible_submitter(tmp_path):
    staged = _staged_harness(tmp_path)
    (staged / IMPOSSIBLE).write_text(
        'echo "impossible tasks=$ARRAY_TASKS mem=${ARRAY_MEM:-}" >> "$SUBMIT_LOG"\n',
        encoding="utf-8",
    )
    sacct = "#!/bin/bash\nprintf '9_1|TIMEOUT\\n9_2|OUT_OF_MEMORY\\n'\n"
    bin_dir = _bin_dir(tmp_path, sacct=sacct)
    submit_log = tmp_path / "submit.log"
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "WORK_ROOT": str(tmp_path / "work"),
        "GT_MODELS": "a b",
        "RETRY_ARRAY_ID": "9",
        "RETRY_ROUND": "0",
        "SUBMIT_LOG": str(submit_log),
        "RETRY_SUBMIT_SCRIPT": IMPOSSIBLE,
    }
    result = subprocess.run(
        ["bash", str(SLURM_DIR / "holdout_retry.sbatch")],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert submit_log.read_text(encoding="utf-8").splitlines() == [
        "impossible tasks=1 mem=",
        "impossible tasks=2 mem=128G",
    ]


def test_the_retry_job_needs_to_be_told_which_sweep_it_retries(tmp_path):
    _staged_harness(tmp_path)
    env = {
        "PATH": os.environ["PATH"], "HOME": str(tmp_path), "WORK_ROOT": str(tmp_path / "work"),
        "GT_MODELS": "a b", "RETRY_ARRAY_ID": "9",
    }
    result = subprocess.run(
        ["bash", str(SLURM_DIR / "holdout_retry.sbatch")],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0 and "RETRY_SUBMIT_SCRIPT" in result.stderr
