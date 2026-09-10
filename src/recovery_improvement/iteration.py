"""One iteration of a recovery-improvement campaign.

An iteration is a directory ``<campaign root>/iter<N>/`` and goes:

1. clone the subject repo from the previous iteration's branch (iteration 1:
   from the campaign's source checkout) onto branch ``branch_name(N)``;
2. digest the sweeps to review into ``digest.md``;
3. run the review agent with a prompt built from the template, the digest,
   earlier prescriptions and the journal; the agent's deliverables are
   ``prescription.md``, commits on the branch, and exactly one of
   ``next_run.env`` / ``STOP``;
4. validate the deliverables — repair up to ``max_review_repairs`` times with
   the problems injected into the prompt, then fail loudly;
5. record the declared sweep. By default (``auto_launch=False``) nothing is
   submitted: the user reads the prescription and launches the sweep with
   ``launch_next.sh`` when they decide to. With ``auto_launch=True`` the sweep
   is launched from the iteration's repo and the next review job is chained
   on its analysis job (the last iteration chains a ``finalize`` job).

The agent launcher and the two Slurm submissions are injected callables so
the whole flow is testable without Slurm or an LLM; ``slurm.py`` holds the
real ones.
"""

from __future__ import annotations

import json
import string
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Union

from src.recovery_improvement.campaign import Campaign
from src.recovery_improvement.digest import build_digest
from src.recovery_improvement.next_run import (
    NextRun,
    describe_allowed_keys,
    parse_next_run,
    sweep_env,
)
from src.recovery_improvement.session_limit import (
    SessionLimitHit,
    detect_session_limit,
    detect_unusable_model,
)

REPO_DIRNAME = "repo"
SWEEP_DIRNAME = "sweep"
DIGEST_NAME = "digest.md"
PROMPT_NAME = "prompt.md"
PRESCRIPTION_NAME = "prescription.md"
NEXT_RUN_NAME = "next_run.env"
STOP_NAME = "STOP"
JOBS_NAME = "jobs.json"
AGENT_LOG_NAME = "claude_stream.jsonl"
FINAL_DIGEST_NAME = "final_digest.md"

MODE_REVIEW = "review"
MODE_FINALIZE = "finalize"


@dataclass(frozen=True)
class SweepJobs:
    """Slurm job ids of one launched sweep (setup -> array -> analysis)."""

    setup_id: str
    array_id: str
    analysis_id: str


@dataclass
class IterationResult:
    iteration: int
    decision: str
    """``sweep`` (a sweep was launched) or ``stop`` (the agent ended the campaign)."""
    repairs_used: int
    review_success: bool
    sweep_jobs: Optional[SweepJobs] = None
    next_review_job: Optional[str] = None
    stop_reason: str = ""
    sweep_env: dict[str, str] = field(default_factory=dict)
    """The launcher env the declared sweep resolves to (defaults + overrides)."""
    launch_command: str = ""
    """How the user launches the declared sweep when auto_launch is off."""


RunAgent = Callable[..., tuple[bool, str]]
"""``run_agent(prompt, *, cwd, log_path) -> (success, result_text)``."""
SubmitSweep = Callable[[Path, dict[str, str]], SweepJobs]
"""``submit_sweep(repo, env) -> SweepJobs``."""
SubmitReview = Callable[[int, Optional[str], str], str]
"""``submit_review(iteration, after_job_id, mode) -> job id``."""


def git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd``; raise with its output on failure."""
    proc = subprocess.run(
        ["git", *args], cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {cwd} (exit {proc.returncode}):\n"
            f"{proc.stdout}{proc.stderr}"
        )
    return proc.stdout.strip()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def append_journal(campaign: Campaign, text: str) -> None:
    with campaign.journal_path.open("a", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n\n")


def roots_to_review(campaign: Campaign, iteration: int) -> list[tuple[str, Path]]:
    """(label, root) pairs the iteration digests, oldest first.

    Iteration 1 reviews the campaign's baseline sweeps. Later iterations
    review the reference baseline plus every earlier iteration's sweep, so the
    agent sees the whole trajectory of its own changes.
    """
    if iteration == 1:
        return [(root.name, root) for root in campaign.baseline_roots]
    roots = [(campaign.baseline_roots[0].name, campaign.baseline_roots[0])]
    for k in range(1, iteration):
        roots.append((f"iter{k}", campaign.iteration_dir(k) / SWEEP_DIRNAME))
    return roots


def primary_label(campaign: Campaign, iteration: int) -> str:
    return campaign.baseline_roots[0].name if iteration == 1 else f"iter{iteration - 1}"


def prepare_repo(campaign: Campaign, iteration: int) -> Path:
    """Clone the previous iteration's branch (or the source checkout) and
    branch it for this iteration. A re-run reuses an existing clone that is
    already on the iteration's branch and fails on anything else."""
    iter_dir = campaign.iteration_dir(iteration)
    repo = iter_dir / REPO_DIRNAME
    branch = campaign.branch_name(iteration)
    if iteration == 1:
        source, source_branch = campaign.source_repo, campaign.base_branch
    else:
        source = campaign.iteration_dir(iteration - 1) / REPO_DIRNAME
        source_branch = campaign.branch_name(iteration - 1)
    if not (source / ".git").exists():
        raise FileNotFoundError(f"cannot branch iteration {iteration}: {source} is not a git repo")

    if repo.exists():
        current = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        if current != branch:
            raise RuntimeError(
                f"{repo} exists but is on branch {current!r}, not {branch!r}; "
                "remove it (or use a fresh campaign) before re-running this iteration"
            )
        return repo

    iter_dir.mkdir(parents=True, exist_ok=True)
    git(iter_dir, "clone", "--quiet", "--branch", source_branch, "--", str(source), str(repo))
    git(repo, "checkout", "-q", "-b", branch)
    return repo


def previous_prescriptions(campaign: Campaign, iteration: int) -> str:
    parts = []
    for k in range(1, iteration):
        path = campaign.iteration_dir(k) / PRESCRIPTION_NAME
        body = path.read_text(encoding="utf-8").strip() if path.is_file() else "(missing)"
        parts.append(f"### Iteration {k} prescription\n\n{body}")
    return "\n\n".join(parts) if parts else "(none — this is the first iteration)"


def compose_prompt(
    template: str,
    *,
    campaign: Campaign,
    iteration: int,
    repo: Path,
    digest: str,
    repair_feedback: str = "",
    venv_py: str = "python",
    plan: str = "",
) -> str:
    """Fill the prompt template (``$placeholder`` syntax; unknown ones raise).
    ``plan`` is a review panel's consensus plan, when one is handed over."""
    journal = (
        campaign.journal_path.read_text(encoding="utf-8").strip()
        if campaign.journal_path.is_file()
        else ""
    )
    mapping = {
        "campaign_name": campaign.name,
        "campaign_root": str(campaign.root),
        "iteration": str(iteration),
        "max_iterations": str(campaign.max_iterations),
        "iter_dir": str(campaign.iteration_dir(iteration)),
        "repo": str(repo),
        "branch": campaign.branch_name(iteration),
        "source_repo": str(campaign.source_repo),
        "base_branch": campaign.base_branch,
        "venv_py": venv_py,
        "sweep_defaults": "\n".join(f"{k}={v}" for k, v in campaign.sweep_defaults.items()),
        "allowed_keys": describe_allowed_keys(),
        "primary_label": primary_label(campaign, iteration),
        "digest": digest,
        "previous_prescriptions": previous_prescriptions(campaign, iteration),
        "journal": journal or "(empty)",
        "repair_feedback": repair_feedback,
        "plan": plan.strip() or "(none — no panel plan was handed over; work from the evidence)",
    }
    try:
        return string.Template(template).substitute(mapping)
    except KeyError as exc:
        raise ValueError(f"prompt template uses an unknown placeholder: {exc}") from exc


def check_deliverables(iter_dir: Path, repo: Path, campaign: Campaign) -> list[str]:
    """Every way the agent's deliverables fall short of the contract."""
    problems: list[str] = []
    prescription = iter_dir / PRESCRIPTION_NAME
    if not prescription.is_file() or not prescription.read_text(encoding="utf-8").strip():
        problems.append(f"{PRESCRIPTION_NAME} is missing or empty at {prescription}")

    next_run = iter_dir / NEXT_RUN_NAME
    stop = iter_dir / STOP_NAME
    if next_run.is_file() and stop.is_file():
        problems.append(f"both {NEXT_RUN_NAME} and {STOP_NAME} exist in {iter_dir}; write exactly one")
    elif not next_run.is_file() and not stop.is_file():
        problems.append(
            f"neither {NEXT_RUN_NAME} nor {STOP_NAME} exists in {iter_dir}; write {NEXT_RUN_NAME} "
            f"to launch a sweep (an empty file = the campaign defaults) or {STOP_NAME} with a reason"
        )
    elif next_run.is_file():
        try:
            parse_next_run(next_run, repo=repo, defaults=campaign.sweep_defaults)
        except ValueError as exc:
            problems.append(str(exc))

    dirty = git(repo, "status", "--porcelain")
    if dirty:
        shown = "\n".join(dirty.splitlines()[:10])
        problems.append(
            f"uncommitted changes in {repo} (commit them on the branch, or delete them; "
            f"the sweep copies the working tree, so anything uncommitted is unreproducible):\n{shown}"
        )
    return problems


def read_decision(iter_dir: Path, repo: Path, campaign: Campaign) -> tuple[str, Union[NextRun, str]]:
    """``("stop", reason)`` or ``("sweep", NextRun)``; call after validation."""
    stop = iter_dir / STOP_NAME
    if stop.is_file():
        return "stop", stop.read_text(encoding="utf-8").strip() or "(no reason given)"
    return "sweep", parse_next_run(
        iter_dir / NEXT_RUN_NAME, repo=repo, defaults=campaign.sweep_defaults
    )


# Artefacts of an attempt that get renamed ``<stem>.attempt<k><suffix>`` when
# the iteration is resumed, so no transcript or brief is overwritten.
_ATTEMPT_ARTEFACTS = (AGENT_LOG_NAME, PROMPT_NAME, "claude_stream.repair1.jsonl", "prompt.repair1.md")


def archive_previous_attempt(iter_dir: Path, repo: Path) -> str:
    """If the iteration was interrupted (a session ran but never completed),
    keep the previous attempt's transcript/brief under ``*.attempt<k>.*`` and
    return a note for the new session describing what it left behind.
    Returns "" for a fresh iteration."""
    if not (iter_dir / AGENT_LOG_NAME).exists():
        return ""
    attempt = 1 + len(list(iter_dir.glob("claude_stream.attempt*.jsonl")))
    for name in _ATTEMPT_ARTEFACTS:
        path = iter_dir / name
        if path.exists():
            stem, suffix = name.split(".", 1)
            path.rename(iter_dir / f"{stem}.attempt{attempt}.{suffix}")
    present = [
        name for name in (PRESCRIPTION_NAME, NEXT_RUN_NAME, STOP_NAME)
        if (iter_dir / name).exists()
    ]
    scratch = iter_dir / "scratch"
    scratch_files = sorted(p.name for p in scratch.iterdir()) if scratch.is_dir() else []
    dirty = git(repo, "status", "--porcelain")
    commits = git(repo, "log", "--oneline", "-10")
    return (
        f"## RESUMING AN INTERRUPTED SESSION (attempt {attempt + 1})\n\n"
        "A previous session of you worked on this iteration but was cut off before the "
        "deliverables were complete (e.g. by the subscription's session limit). Its work is "
        "still here — do not start over. Read what exists, then finish the contract "
        "(prescription.md, committed clean tree, next_run.env or STOP, journal entry).\n\n"
        f"- previous transcript: `{iter_dir / f'claude_stream.attempt{attempt}.jsonl'}` "
        "(stream-json; grep the `text` fields for its reasoning)\n"
        f"- deliverable files already present: {', '.join(present) if present else 'none'}\n"
        f"- scratch files: {', '.join(scratch_files) if scratch_files else 'none'}\n"
        f"- `git status --porcelain` in the repo:\n```\n{dirty or '(clean)'}\n```\n"
        f"- recent commits on the branch:\n```\n{commits}\n```\n"
    )


def _raise_if_unusable_model(campaign: Campaign, result_text: str) -> None:
    """Stop at once if the configured review model is unavailable.

    No repair round can fix a model the login cannot use, and burning one
    costs a session and buries the real message under a deliverables error.
    """
    model = detect_unusable_model(result_text)
    if model is None:
        return
    raise RuntimeError(
        f"the review model {model!r} is not available to this login "
        f"(claude reported: {result_text.strip()[:200]}). Set REVIEW_MODEL in "
        f"{campaign.root / 'campaign.env'} to a model the login can use, then "
        "resubmit this iteration; its partial work is kept and resumed."
    )


def _raise_if_session_limit(campaign: Campaign, result_text: str) -> None:
    """The subscription's session limit ends the session with a message in the
    result text (Claude still reports subtype "success"). Stop here — no repair
    round, nothing launched — so the CLI can requeue the iteration for after
    the reset; the resume path picks the partial work up."""
    limit = detect_session_limit(result_text)
    if limit is None:
        return
    when = limit.reset_at.isoformat() if limit.reset_at else "unknown"
    append_journal(campaign, f"- session limit hit: {limit.message} (reset at {when}); iteration paused")
    raise SessionLimitHit(limit)


def _repair_feedback(problems: list[str], attempt: int) -> str:
    bullets = "\n".join(f"- {p}" for p in problems)
    return (
        f"## REPAIR REQUIRED (attempt {attempt})\n\n"
        "Your previous session in this iteration ended without valid deliverables. "
        "Your repo, commits and files are still in place; fix exactly these problems "
        "and finish the contract:\n\n" + bullets + "\n"
    )


def launch_command(campaign: Campaign, iteration: int) -> str:
    script = campaign.source_repo / "scripts" / "recovery_improvement" / "launch_next.sh"
    return f"bash {script} {campaign.name} {iteration}"


def run_iteration(
    campaign: Campaign,
    iteration: int,
    *,
    run_agent: RunAgent,
    prompt_template: str,
    auto_launch: bool = False,
    submit_sweep: Optional[SubmitSweep] = None,
    submit_review: Optional[SubmitReview] = None,
    venv_py: str = "python",
    plan_text: str = "",
) -> IterationResult:
    if auto_launch and (submit_sweep is None or submit_review is None):
        raise ValueError("auto_launch=True needs both submit_sweep and submit_review")
    if campaign.stop_path.exists():
        raise RuntimeError(f"campaign STOP file present at {campaign.stop_path}; not running")
    if not 1 <= iteration <= campaign.max_iterations:
        raise ValueError(
            f"iteration {iteration} is outside 1..{campaign.max_iterations} (MAX_ITERATIONS)"
        )
    iter_dir = campaign.iteration_dir(iteration)
    if (iter_dir / JOBS_NAME).exists():
        raise RuntimeError(
            f"{iter_dir / JOBS_NAME} exists: iteration {iteration} already completed. "
            "Start the next iteration instead, or a new campaign."
        )
    iter_dir.mkdir(parents=True, exist_ok=True)

    repo = prepare_repo(campaign, iteration)
    resume_note = archive_previous_attempt(iter_dir, repo)
    roots = roots_to_review(campaign, iteration)
    digest = build_digest(roots, primary_label=primary_label(campaign, iteration))
    (iter_dir / DIGEST_NAME).write_text(digest, encoding="utf-8")

    append_journal(
        campaign,
        f"## Iteration {iteration}{' (resumed)' if resume_note else ''} — {_timestamp()}\n\n"
        + "\n".join(f"- reviewed `{label}`: `{root}`" for label, root in roots)
        + f"\n- repo: `{repo}` (branch `{campaign.branch_name(iteration)}`)",
    )

    prompt = compose_prompt(
        prompt_template, campaign=campaign, iteration=iteration, repo=repo,
        digest=digest, repair_feedback=resume_note, venv_py=venv_py, plan=plan_text,
    )
    (iter_dir / PROMPT_NAME).write_text(prompt, encoding="utf-8")
    success, result_text = run_agent(prompt, cwd=repo, log_path=iter_dir / AGENT_LOG_NAME)
    _raise_if_unusable_model(campaign, result_text)
    _raise_if_session_limit(campaign, result_text)

    problems = check_deliverables(iter_dir, repo, campaign)
    repairs = 0
    while problems and repairs < campaign.max_review_repairs:
        repairs += 1
        prompt = compose_prompt(
            prompt_template, campaign=campaign, iteration=iteration, repo=repo,
            digest=digest, repair_feedback=_repair_feedback(problems, repairs),
            venv_py=venv_py, plan=plan_text,
        )
        (iter_dir / f"prompt.repair{repairs}.md").write_text(prompt, encoding="utf-8")
        success, result_text = run_agent(
            prompt, cwd=repo, log_path=iter_dir / f"claude_stream.repair{repairs}.jsonl"
        )
        _raise_if_session_limit(campaign, result_text)
        problems = check_deliverables(iter_dir, repo, campaign)
    if problems:
        raise RuntimeError(
            f"iteration {iteration} deliverables are invalid after {repairs} repair(s); "
            "nothing was launched:\n" + "\n".join(f"- {p}" for p in problems)
        )

    decision, payload = read_decision(iter_dir, repo, campaign)
    result = IterationResult(
        iteration=iteration, decision=decision, repairs_used=repairs, review_success=success,
    )
    if decision == "stop":
        result.stop_reason = str(payload)
        append_journal(
            campaign,
            f"- review agent: {'ok' if success else 'ended abnormally'} ({repairs} repair(s))\n"
            f"- decision: STOP — {result.stop_reason}",
        )
    else:
        next_run = payload
        assert isinstance(next_run, NextRun)
        env = sweep_env(next_run, campaign, repo=repo, work_root=iter_dir / SWEEP_DIRNAME)
        result.sweep_env = env
        overrides = ", ".join(f"{k}={v}" for k, v in next_run.values.items()) or "campaign defaults"
        entry = (
            f"- review agent: {'ok' if success else 'ended abnormally'} ({repairs} repair(s))\n"
            f"- decision: sweep — {next_run.note or '(no note)'}\n"
            f"- sweep overrides: {overrides}\n"
        )
        if auto_launch:
            assert submit_sweep is not None and submit_review is not None
            result.sweep_jobs = submit_sweep(repo, env)
            next_iteration = iteration + 1
            mode = MODE_REVIEW if next_iteration <= campaign.max_iterations else MODE_FINALIZE
            result.next_review_job = submit_review(
                next_iteration, result.sweep_jobs.analysis_id, mode
            )
            entry += (
                f"- sweep jobs: setup {result.sweep_jobs.setup_id}, array "
                f"{result.sweep_jobs.array_id}, analysis {result.sweep_jobs.analysis_id}; "
                f"output `{iter_dir / SWEEP_DIRNAME}`\n"
                f"- next: {mode} job {result.next_review_job} (iteration {next_iteration}), "
                f"after analysis {result.sweep_jobs.analysis_id}"
            )
        else:
            result.launch_command = launch_command(campaign, iteration)
            entry += (
                f"- NOT launched (auto-launch is off). Read `{iter_dir / PRESCRIPTION_NAME}`, "
                f"then to run this sweep: `{result.launch_command}` "
                f"(output would land in `{iter_dir / SWEEP_DIRNAME}`)"
            )
        append_journal(campaign, entry)
    (iter_dir / JOBS_NAME).write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def finalize_campaign(campaign: Campaign, iteration: int) -> Path:
    """After the last sweep: digest every sweep into ``final_digest.md``."""
    roots = roots_to_review(campaign, iteration)
    digest = build_digest(roots, primary_label=primary_label(campaign, iteration))
    out = campaign.root / FINAL_DIGEST_NAME
    out.write_text(digest, encoding="utf-8")
    append_journal(
        campaign,
        f"## Campaign complete — {_timestamp()}\n\n"
        f"- all {campaign.max_iterations} iteration(s) ran; final digest: `{out}`\n"
        f"- branches: " + ", ".join(
            f"`{campaign.branch_name(k)}` in `{campaign.iteration_dir(k) / REPO_DIRNAME}`"
            for k in range(1, iteration)
        ),
    )
    return out
