"""The real adapters ``run_iteration`` is injected with: the Claude review
agent, the sweep launch, and the chained review submission.

Everything that talks to Slurm or spawns Claude lives here so
:mod:`iteration` stays testable; the pure parts (output parsing, environment
scrubbing, argv construction) are unit-tested directly.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Mapping, Optional

from src.recovery_improvement.campaign import Campaign
from src.recovery_improvement.iteration import RunAgent, SweepJobs
from src.recovery_improvement.next_run import ALLOWED_KEYS
from src.runtime import token_usage
from src.runtime.coding_agent import run_coding_agent

SUBMIT_SCRIPT_REL = Path("scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh")
_SUBMITTED = re.compile(r"^submitted (setup|array|analysis) job:\s+(\d+)", re.MULTILINE)

# Env-var prefixes / names that must not leak from the review job into the
# sweep or the next review job: Slurm's own per-job state, uv's env pins, the
# review job's caches and locations, and every sweep knob (so a stale value
# from this shell can't override the campaign defaults).
_SCRUB_PREFIXES = ("SLURM_", "SBATCH_", "SRUN_", "UV_", "XDG_")
_SCRUB_NAMES = frozenset({
    "VENV_PY", "WORK_ROOT", "REPO", "PYTENSOR_FLAGS", "ITERATION", "MODE",
    "CAMPAIGN_ROOT", "HOLDOUT_SLURM_DIR", "N_GTS", "KEEP_REPO_COPY", "ARRAY_TASKS",
    "GT_MODELS_SRC", "GT_FAMILY_SRC", "INODE_CEIL_PCT", "SMOKE_TASKS",
    *ALLOWED_KEYS,
})

# Tools the review agent must never use, enforced by Claude Code on top of the
# prompt's rules: it may not cancel or alter Slurm jobs, and never pushes.
REVIEW_DISALLOWED_TOOLS = (
    "Bash(scancel:*)",
    "Bash(scontrol update:*)",
    "Bash(scontrol hold:*)",
    "Bash(scontrol release:*)",
    "Bash(scontrol requeue:*)",
    "Bash(git push:*)",
)

Runner = Callable[..., subprocess.CompletedProcess]


def scrubbed_environment(env: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in env.items()
        if not key.startswith(_SCRUB_PREFIXES) and key not in _SCRUB_NAMES
    }


def parse_submit_output(text: str) -> SweepJobs:
    """Job ids from submit_holdout_test_retest.sh's stdout; raise if any missing."""
    found = {stage: job_id for stage, job_id in _SUBMITTED.findall(text)}
    missing = [stage for stage in ("setup", "array", "analysis") if stage not in found]
    if missing:
        raise RuntimeError(
            f"sweep submit output has no job id for stage(s) {missing}:\n{text}"
        )
    return SweepJobs(
        setup_id=found["setup"], array_id=found["array"], analysis_id=found["analysis"]
    )


def submit_sweep(repo: Path, env: dict[str, str], *, runner: Runner = subprocess.run) -> SweepJobs:
    """Launch a sweep from ``repo`` via its own submit script.

    The script and the Slurm scripts it chains run from ``repo`` (the
    iteration's branch), so the agent's edits to them take effect. Dependent
    jobs are submitted with kill-on-invalid-dependency so a failed setup job
    cannot leave the chain pending forever.
    """
    script = repo / SUBMIT_SCRIPT_REL
    if not script.is_file():
        raise FileNotFoundError(f"sweep submit script missing from the iteration repo: {script}")
    full_env = {**scrubbed_environment(os.environ), **env, "SBATCH_KILL_INVALID_DEP": "yes"}
    proc = runner(
        ["bash", str(script)], cwd=repo, env=full_env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sweep submission failed (exit {proc.returncode}) for {script}:\n{proc.stdout}"
        )
    print(proc.stdout)
    return parse_submit_output(proc.stdout)


def review_sbatch_command(
    campaign: Campaign,
    iteration: int,
    after_job_id: Optional[str],
    mode: str,
    *,
    begin: Optional[str] = None,
) -> list[str]:
    """argv that submits the review (or finalize) job for ``iteration``.
    ``begin`` (``sbatch --begin``) delays the start, e.g. until a session
    limit resets."""
    slurm = campaign.review_slurm
    log_dir = campaign.root / "slurm_logs"
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=recovery_review_{campaign.name}",
        f"--partition={slurm['REVIEW_PARTITION']}",
        f"--time={slurm['REVIEW_TIME']}",
        f"--cpus-per-task={slurm['REVIEW_CPUS']}",
        f"--mem={slurm['REVIEW_MEM']}",
        f"--output={log_dir}/{mode}_iter{iteration}_%j.out",
        f"--error={log_dir}/{mode}_iter{iteration}_%j.out",
        # The chain has no watcher: if a review job dies for a reason the
        # session-limit requeue does not cover (walltime, node failure, a bad
        # sbatch), nothing downstream runs and nothing says so. Slurm mails the
        # job owner, which survives this process exiting. No MailUser: Slurm
        # defaults to the submitting account.
        "--mail-type=FAIL,TIMEOUT",
        f"--export=ALL,CAMPAIGN_ROOT={campaign.root},ITERATION={iteration},MODE={mode}",
    ]
    if after_job_id:
        # afterany: the review must run even when the sweep failed — diagnosing
        # a broken sweep is exactly the agent's job.
        cmd.append(f"--dependency=afterany:{after_job_id}")
    if begin:
        cmd.append(f"--begin={begin}")
    cmd.append(str(campaign.driver_sbatch))
    return cmd


def submit_review(
    campaign: Campaign,
    iteration: int,
    after_job_id: Optional[str],
    mode: str,
    *,
    begin: Optional[str] = None,
    runner: Runner = subprocess.run,
) -> str:
    if not campaign.driver_sbatch.is_file():
        raise FileNotFoundError(f"review sbatch script not found: {campaign.driver_sbatch}")
    (campaign.root / "slurm_logs").mkdir(parents=True, exist_ok=True)
    cmd = review_sbatch_command(campaign, iteration, after_job_id, mode, begin=begin)
    proc = runner(
        cmd, env=scrubbed_environment(os.environ), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed (exit {proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    job_id = proc.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch --parsable returned no job id: {proc.stdout!r}")
    return job_id


def make_run_agent(campaign: Campaign, iteration: int) -> RunAgent:
    """The Claude Code review session, with the campaign's caps applied."""
    extra_args = [
        "--max-turns", str(campaign.review_max_turns),
        "--max-budget-usd", str(campaign.review_max_budget_usd),
        "--disallowedTools", *REVIEW_DISALLOWED_TOOLS,
    ]
    allowed_dirs = [campaign.root, *campaign.baseline_roots]

    def run(prompt: str, *, cwd: Path, log_path: Path) -> tuple[bool, str]:
        token_usage.start_usage_log(log_path.parent / "token_usage.jsonl")
        return run_coding_agent(
            prompt,
            cwd=cwd,
            log_path=log_path,
            allowed_dirs=allowed_dirs,
            model=campaign.review_model,
            timeout_secs=campaign.review_timeout_sec,
            backend="claude",
            usage_label=f"recovery_improvement:iter{iteration}",
            extra_args=extra_args,
        )

    return run
