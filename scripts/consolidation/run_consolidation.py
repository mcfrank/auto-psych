"""CLI: drive the consolidation plan, one Claude session per phase.

Called by ``consolidate.sbatch`` on a compute node after it has prepared the
clone and the venv. Loops over the plan's phases: for each phase not yet done
it composes the brief, runs a Claude Code session in the clone, validates the
phase marker (one repair session on an invalid one), and moves on. It
requeues itself — never sleeps — when

* the subscription's session limit is hit (``--begin`` at the reset),
* the remaining walltime cannot fit another session, or
* a phase submitted Slurm jobs (P7 smoke, P9 sweep, P10 evaluation) and the
  next phase must wait for them (``--dependency=afterany:<ids>``).

Usage (inside the sbatch job)::

    $VENV_PY scripts/consolidation/run_consolidation.py --work-root $WORK_ROOT

``--dry-run`` writes the next phase's brief to ``<work_root>/dryrun/`` and
exits without spawning anything.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.consolidation.driver import (  # noqa: E402
    BRANCH,
    PLAN_REL,
    PROGRESS_DIRNAME,
    Blocked,
    Phase,
    compose_prompt,
    disallowed_tools,
    jobs_file_owner,
    jobs_still_queued,
    load_env,
    needs_walltime_requeue,
    next_phase,
    parse_jobs_file,
    phase_round,
    render_inputs,
    requeue_command,
    seconds_left,
    validate_done_marker,
)
from src.recovery_improvement.session_limit import (  # noqa: E402
    detect_session_limit,
    detect_unusable_model,
    retry_begin_time,
)
from src.recovery_improvement.slurm import scrubbed_environment  # noqa: E402
from src.runtime import token_usage  # noqa: E402
from src.runtime.coding_agent import run_coding_agent  # noqa: E402

INPUT_KEYS = (
    "SOURCE_REPO", "MAIN_SHA", "ITER3_REPO", "ITER3_SHA", "ITER4_SHA", "ITER5_SHA",
    "ARMC_SHA", "LEAKAGE_PATCH", "LEAKAGE_PATCH_SHA256", "ARMC_RUN_ROOT",
    "ITER3_SWEEP", "ITER2_SWEEP", "BASELINE_SWEEP",
    "SWEEP_ARMS", "SWEEP_N_REPEATS", "SWEEP_BASE_SEED", "SWEEP_MAX_PARALLEL",
)
WALLTIME_MARGIN_SEC = 900
MAX_LIMIT_WAITS = 12
EXIT_BLOCKED = 3


@dataclass
class Args:
    """Drive the consolidation plan inside its Slurm job."""

    work_root: Path
    """The job's work root (holds consolidation.env, repo/, progress/, venv/)."""
    dry_run: bool = False
    """Write the next phase's brief to <work_root>/dryrun/ and exit."""


def git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {cwd} (exit {proc.returncode}):\n"
            f"{proc.stdout}{proc.stderr}"
        )
    return proc.stdout.strip()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def append_status(work_root: Path, text: str) -> None:
    with (work_root / "STATUS.md").open("a", encoding="utf-8") as fh:
        fh.write(f"- {_stamp()} — {text.rstrip()}\n")


class Driver:
    def __init__(self, work_root: Path) -> None:
        self.work_root = work_root
        self.env = load_env(work_root / "consolidation.env")
        self.repo = work_root / "repo"
        self.progress = work_root / PROGRESS_DIRNAME
        self.progress.mkdir(parents=True, exist_ok=True)
        self.venv_py = os.environ.get("VENV_PY") or sys.executable
        self.template = (
            Path(self.env["SOURCE_REPO"]) / "scripts" / "consolidation" / "phase_prompt.md"
        ).read_text(encoding="utf-8")
        plan_path = Path(self.env["SOURCE_REPO"]) / PLAN_REL
        if not plan_path.is_file():
            raise FileNotFoundError(f"plan not found: {plan_path}")
        self.plan = plan_path.read_text(encoding="utf-8")

    # --- Slurm ------------------------------------------------------------------

    def _seconds_left(self) -> Optional[int]:
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return None
        proc = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%L"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RuntimeError(f"squeue could not report time left for job {job_id}: {proc.stderr}")
        return seconds_left(proc.stdout.strip().splitlines()[0])

    def _still_queued(self, job_ids: list[str]) -> set[str]:
        proc = subprocess.run(
            ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        # squeue exits non-zero when none of the ids exist any more; that is
        # "nothing queued", not an error — but an unknown failure is.
        if proc.returncode != 0 and "Invalid job id" not in proc.stderr:
            raise RuntimeError(f"squeue failed: {proc.stderr}")
        return jobs_still_queued(proc.stdout) & set(job_ids)

    def requeue(self, *, dependency: Optional[str] = None, begin: Optional[str] = None, why: str) -> str:
        cmd = requeue_command(
            Path(self.env["DRIVER_SBATCH"]),
            job_name=self.env["JOB_NAME"],
            partition=self.env["PARTITION"],
            time=self.env["TIME"],
            cpus=self.env["CPUS"],
            mem=self.env["MEM"],
            log_dir=self.work_root / "slurm_logs",
            export=f"ALL,WORK_ROOT={self.work_root}",
            dependency=dependency,
            begin=begin,
        )
        proc = subprocess.run(
            cmd, env=scrubbed_environment(os.environ), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"requeue failed (exit {proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
        job_id = proc.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"sbatch --parsable returned no job id: {proc.stdout!r}")
        append_status(self.work_root, f"requeued as job {job_id} ({why})")
        print(f"[driver] requeued as job {job_id}: {why}")
        return job_id

    def _wait_for(self, jobs_file: str) -> bool:
        """Requeue ``afterany`` the ids in ``jobs_file`` that are still queued.
        Returns True if a requeue was submitted (the caller must exit)."""
        owner = jobs_file_owner(jobs_file)
        job_ids = parse_jobs_file(self.progress / jobs_file, owner.required_labels)
        queued = self._still_queued(job_ids)
        if not queued:
            return False
        self.requeue(
            dependency="afterany:" + ":".join(sorted(queued)),
            why=f"waiting for {jobs_file} jobs {sorted(queued)}",
        )
        return True

    # --- one phase ----------------------------------------------------------------

    def brief(self, phase: Phase, *, resume_note: str = "", repair_feedback: str = "") -> str:
        mapping = {
            "phase_id": phase.id,
            "phase_title": phase.title,
            "repo": str(self.repo),
            "branch": BRANCH,
            "source_repo": self.env["SOURCE_REPO"],
            "progress_dir": str(self.progress),
            "work_root": str(self.work_root),
            "venv_py": self.venv_py,
            "inputs": render_inputs(self.env, INPUT_KEYS),
            "armc_run_root": self.env["ARMC_RUN_ROOT"],
            "resume_note": resume_note,
            "repair_feedback": repair_feedback,
            "plan": self.plan,
        }
        return compose_prompt(self.template, mapping)

    def _resume_note(self, phase: Phase) -> str:
        transcripts = sorted(self.progress.glob(f"{phase.id}.session*.jsonl"))
        if not transcripts:
            return ""
        dirty = git(self.repo, "status", "--porcelain")
        commits = git(self.repo, "log", "--oneline", "-12")
        retries = sorted(self.progress.glob(f"{phase.id}.retry*"))
        retry_text = ""
        if retries:
            reasons = "\n".join(
                f"  - {r.name}: {r.read_text(encoding='utf-8').strip()[:200]}" for r in retries
            )
            retry_text = (
                f"- this phase was RE-OPENED by a later phase (round {phase_round(self.progress, phase.id)}); "
                f"its retry markers say why:\n{reasons}\n"
            )
        return (
            f"## RESUMING PHASE {phase.id} (session {len(transcripts) + 1})\n\n"
            "A previous session of you worked on this phase and either was cut off "
            "(session limit or walltime) or the phase was re-opened. Its work is still "
            "in the clone — do not start over. Read what exists, then finish the phase.\n\n"
            f"- previous transcripts: {', '.join(str(t) for t in transcripts)} "
            "(stream-json; grep the `text` fields)\n"
            f"{retry_text}"
            f"- `git status --porcelain`:\n```\n{dirty or '(clean)'}\n```\n"
            f"- recent commits:\n```\n{commits}\n```\n"
        )

    def _run_session(self, phase: Phase, prompt: str, label: str) -> tuple[bool, str]:
        n = len(list(self.progress.glob(f"{phase.id}.session*.jsonl"))) + 1
        log_path = self.progress / f"{phase.id}.session{n}.jsonl"
        (self.progress / f"{phase.id}.session{n}.prompt.md").write_text(prompt, encoding="utf-8")
        token_usage.start_usage_log(self.progress / "token_usage.jsonl")
        extra_args = [
            "--max-turns", self.env["MAX_TURNS"],
            "--max-budget-usd", self.env["MAX_BUDGET_USD"],
            "--disallowedTools", *disallowed_tools(phase),
        ]
        append_status(self.work_root, f"{phase.id} session {n} started ({label})")
        success, result_text = run_coding_agent(
            prompt,
            cwd=self.repo,
            log_path=log_path,
            allowed_dirs=[
                self.work_root,
                Path(self.env["ARMC_RUN_ROOT"]),
                Path(self.env["ITER2_SWEEP"]),
                Path(self.env["ITER3_SWEEP"]),
                Path(self.env["BASELINE_SWEEP"]),
            ],
            model=self.env["MODEL"],
            timeout_secs=int(self.env["TIMEOUT_SEC"]),
            backend="claude",
            usage_label=f"consolidation:{phase.id}",
            extra_args=extra_args,
        )
        append_status(
            self.work_root,
            f"{phase.id} session {n} ended: {'ok' if success else 'abnormal'} — "
            f"{result_text.strip()[:160]!r}",
        )
        return success, result_text

    def _handle_limits(self, result_text: str) -> bool:
        """Requeue on a session limit (returns True); raise on an unusable model."""
        model = detect_unusable_model(result_text)
        if model is not None:
            raise RuntimeError(
                f"model {model!r} is not available to this login (claude reported: "
                f"{result_text.strip()[:200]}). Set MODEL in "
                f"{self.work_root / 'consolidation.env'} and resubmit."
            )
        limit = detect_session_limit(result_text)
        if limit is None:
            return False
        counter = self.progress / "limit_waits"
        waits = int(counter.read_text()) + 1 if counter.is_file() else 1
        if waits > MAX_LIMIT_WAITS:
            raise SystemExit(
                f"session limit hit {waits} times; not requeueing again. Last: {limit.message}"
            )
        counter.write_text(str(waits))
        self.requeue(begin=retry_begin_time(limit), why=f"session limit: {limit.message[:80]}")
        return True

    def _problems(self, phase: Phase) -> list[str]:
        return validate_done_marker(
            self.progress, phase.id,
            head_sha=git(self.repo, "rev-parse", "HEAD"),
            porcelain=git(self.repo, "status", "--porcelain"),
            work_root=self.work_root,
        )

    def run_phase(self, phase: Phase) -> str:
        """``"done"``, ``"requeued"`` or ``"blocked"``."""
        prompt = self.brief(phase, resume_note=self._resume_note(phase))
        _, result = self._run_session(phase, prompt, "main")
        if self._handle_limits(result):
            return "requeued"
        if (self.progress / f"{phase.id}.blocked").exists():
            return "blocked"
        problems = self._problems(phase)
        if problems:
            feedback = (
                "## REPAIR REQUIRED\n\nYour previous session ended without a valid phase "
                "marker. Your commits and files are still in place; fix exactly these "
                "problems and finish the phase:\n\n" + "\n".join(f"- {p}" for p in problems) + "\n"
            )
            prompt = self.brief(phase, resume_note=self._resume_note(phase), repair_feedback=feedback)
            _, result = self._run_session(phase, prompt, "repair")
            if self._handle_limits(result):
                return "requeued"
            if (self.progress / f"{phase.id}.blocked").exists():
                return "blocked"
            problems = self._problems(phase)
        if problems:
            (self.progress / f"{phase.id}.blocked").write_text(
                "driver: deliverables invalid after one repair session:\n"
                + "\n".join(f"- {p}" for p in problems) + "\n",
                encoding="utf-8",
            )
            return "blocked"
        append_status(self.work_root, f"{phase.id} done at {git(self.repo, 'rev-parse', '--short', 'HEAD')}")
        return "done"

    # --- the loop --------------------------------------------------------------------

    def run(self) -> int:
        while True:
            try:
                phase = next_phase(self.progress)
            except Blocked as exc:
                append_status(self.work_root, f"STOPPED — {exc}")
                print(f"[driver] {exc}", file=sys.stderr)
                return EXIT_BLOCKED
            if phase is None:
                append_status(self.work_root, "ALL PHASES DONE — see RESULTS.md")
                print(f"[driver] all phases done; results at {self.work_root / 'RESULTS.md'}")
                return 0

            if phase.waits_for and self._wait_for(phase.waits_for):
                return 0
            if phase.max_rounds is not None and phase_round(self.progress, phase.id) > phase.max_rounds:
                (self.progress / f"{phase.id}.blocked").write_text(
                    f"driver: round {phase_round(self.progress, phase.id)} of {phase.id} exceeds "
                    f"its limit of {phase.max_rounds}; read the retry markers and the last verdict\n",
                    encoding="utf-8",
                )
                continue

            left = self._seconds_left()
            if needs_walltime_requeue(
                seconds_left=left, session_timeout=int(self.env["TIMEOUT_SEC"]),
                margin=WALLTIME_MARGIN_SEC,
            ):
                self.requeue(why=f"only {left}s of walltime left before {phase.id}")
                return 0

            outcome = self.run_phase(phase)
            if outcome == "requeued":
                return 0
            if outcome == "blocked":
                continue  # next_phase raises Blocked with the reason
            if phase.jobs_file:
                job_ids = parse_jobs_file(self.progress / phase.jobs_file, phase.required_labels)
                self.requeue(
                    dependency="afterany:" + ":".join(job_ids),
                    why=f"{phase.id} submitted jobs {job_ids}",
                )
                return 0


def main(args: Args) -> None:
    driver = Driver(args.work_root)
    if args.dry_run:
        phase = next_phase(driver.progress)
        out = driver.work_root / "dryrun"
        out.mkdir(parents=True, exist_ok=True)
        if phase is None:
            print("dry run: all phases done")
            return
        path = out / f"{phase.id}.prompt.md"
        path.write_text(driver.brief(phase), encoding="utf-8")
        print(f"dry run: next phase {phase.id}; brief written to {path}")
        return
    raise SystemExit(driver.run())


if __name__ == "__main__":
    main(tyro.cli(Args))
