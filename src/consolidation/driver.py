"""Pure parts of the consolidation driver.

The consolidation plan (``docs/consolidation_plan_2026_09.md``) is executed as
twenty-two phases, P0..P21, one Claude Code session each. State lives on disk
under ``<work_root>/progress/`` as marker files the agent writes and the
driver validates:

* ``P<k>.done`` — first line ``commit: <sha>`` (must equal HEAD), then a summary;
* ``P<k>.blocked`` — the agent hit a stop condition; the job stops;
* ``<phase.jobs_file>`` — for a phase that submits Slurm jobs (P7, P12 and P20
  smoke, P14 sweep, P15 evaluation): the ids the driver must wait on, per label;
* ``P<k>.retry*`` — a later phase asked for another round of ``P<k>``
  (bounded by ``Phase.max_rounds``).

Everything here is deterministic and unit-tested; the process that spawns
Claude, git and Slurm is ``scripts/consolidation/run_consolidation.py``.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

PLAN_REL = Path("docs/consolidation_plan_2026_09.md")
BRANCH = "consolidate/2026-09"
PROGRESS_DIRNAME = "progress"
BASELINE_FAILING_NAME = "baseline_failing_tests.txt"
BASELINE_COLLECTION_NAME = "baseline_collection_errors.txt"

# Tools the agent must never use, enforced by Claude Code on top of the plan's
# rules. ``sbatch`` is added for every phase that does not submit jobs.
BASE_DISALLOWED_TOOLS = (
    "Bash(scancel:*)",
    "Bash(scontrol update:*)",
    "Bash(scontrol hold:*)",
    "Bash(scontrol release:*)",
    "Bash(scontrol requeue:*)",
    "Bash(git push:*)",
)
SBATCH_TOOL = "Bash(sbatch:*)"


@dataclass(frozen=True)
class Phase:
    id: str
    title: str
    allows_sbatch: bool = False
    jobs_file: Optional[str] = None
    """File under progress/ this phase must write; the driver requeues
    itself ``afterany`` every job id in it."""
    required_labels: tuple[str, ...] = ()
    """Labels (arms) that must be present in ``jobs_file``."""
    waits_for: Optional[str] = None
    """A jobs file an earlier phase wrote; every id in it must have left the
    queue before this phase may start."""
    requires_files: tuple[str, ...] = ()
    """Files under the work root that must exist when the phase is done."""
    max_rounds: Optional[int] = None
    """How many times the phase may run (``P<k>.retry*`` markers re-open it)."""


PHASES: tuple[Phase, ...] = (
    Phase("P0", "Baseline on the untouched base"),
    Phase("P1", "Integrate: merge main, restore the audit, merge arm C"),
    Phase("P2", "Make raw mode genuinely raw end to end"),
    Phase("P3", "Honest metrics and offline diagnostics"),
    Phase("P4", "Reimplement the race presentation"),
    Phase("P5", "Reimplement lens rotation"),
    Phase("P6", "Documentation, decision record, full checks"),
    Phase(
        "P7", "Submit the two smoke chains",
        allows_sbatch=True, jobs_file="smoke_jobs.json",
        required_labels=("featurized", "raw"), max_rounds=2,
    ),
    Phase(
        "P8", "Smoke verdict and handoff",
        waits_for="smoke_jobs.json", requires_files=("VERDICT.md", "HANDOFF.md"),
    ),
    # Amendment of 2026-09-16: raw is the only mode and the feature code must
    # be unreadable by agents (see the plan's amendment before P9).
    Phase("P9", "Raw is the only mode"),
    Phase("P10", "The agents' tree contains no feature code; imports are gated"),
    # Inserted at the user's request (2026-09-16): simplify for human reading
    # before the smoke validates it and the sweep runs on it.
    Phase("P11", "Simplify: make the code readable end to end"),
    Phase(
        "P12", "Submit the smoke cell",
        allows_sbatch=True, jobs_file="isolation_smoke_jobs.json",
        required_labels=("raw",), max_rounds=3,
    ),
    Phase(
        "P13", "Smoke verdict",
        waits_for="isolation_smoke_jobs.json", requires_files=("VERDICT.md", "HANDOFF.md"),
    ),
    Phase(
        "P14", "Launch the 5-repeat recovery sweep",
        allows_sbatch=True, jobs_file="sweep_jobs.json", required_labels=("raw",),
    ),
    Phase(
        "P15", "Submit the RMSE evaluation job",
        waits_for="sweep_jobs.json", allows_sbatch=True,
        jobs_file="analysis_jobs.json", required_labels=("analysis",), max_rounds=2,
    ),
    Phase(
        "P16", "Results: the RMSE evaluation report",
        waits_for="analysis_jobs.json", requires_files=("RESULTS.md",),
    ),
    # Appended at the user's request (2026-09-17): P11's simplification was too
    # conservative, so this runs an aggressive cleanup AFTER the results, when
    # the clone is no longer being copied by pending sweep tasks.
    Phase("P17", "Characterize, then split the oversized modules"),
    Phase("P18", "Reduce the surface"),
    Phase("P19", "Readability"),
    Phase(
        "P20", "Prove it: submit an equivalence smoke",
        allows_sbatch=True, jobs_file="cleanup_smoke_jobs.json",
        required_labels=("raw",), max_rounds=3,
    ),
    Phase(
        "P21", "Cleanup verdict and report",
        waits_for="cleanup_smoke_jobs.json", requires_files=("CLEANUP_REPORT.md",),
    ),
)


class Blocked(RuntimeError):
    """A phase wrote its ``.blocked`` marker: the job must stop, not continue."""


def phase_by_id(phase_id: str) -> Phase:
    for phase in PHASES:
        if phase.id == phase_id:
            return phase
    raise KeyError(f"no consolidation phase {phase_id!r}; known: {[p.id for p in PHASES]}")


def jobs_file_owner(jobs_file: str) -> Phase:
    """The phase that writes ``jobs_file`` (so a waiting phase knows which
    labels the file must carry)."""
    for phase in PHASES:
        if phase.jobs_file == jobs_file:
            return phase
    raise KeyError(f"no phase writes the jobs file {jobs_file!r}")


def disallowed_tools(phase: Phase) -> tuple[str, ...]:
    if phase.allows_sbatch:
        return BASE_DISALLOWED_TOOLS
    return (*BASE_DISALLOWED_TOOLS, SBATCH_TOOL)


# --- progress markers -----------------------------------------------------------


def phase_state(progress_dir: Path, phase_id: str) -> str:
    """``"blocked"`` / ``"done"`` / ``"pending"``. Blocked wins over done."""
    if (progress_dir / f"{phase_id}.blocked").exists():
        return "blocked"
    if (progress_dir / f"{phase_id}.done").exists():
        return "done"
    return "pending"


def next_phase(progress_dir: Path) -> Optional[Phase]:
    """The first phase that is not done, or None when all are. Raises
    :class:`Blocked` (with the reason) if that phase is blocked."""
    for phase in PHASES:
        state = phase_state(progress_dir, phase.id)
        if state == "done":
            continue
        if state == "blocked":
            reason = (progress_dir / f"{phase.id}.blocked").read_text(encoding="utf-8").strip()
            raise Blocked(f"phase {phase.id} is blocked: {reason or '(no reason written)'}")
        return phase
    return None


def phase_round(progress_dir: Path, phase_id: str) -> int:
    """1 for the first run of a phase; +1 per ``P<k>.retry*`` marker a later
    phase wrote to re-open it."""
    return 1 + len(list(progress_dir.glob(f"{phase_id}.retry*")))


def retry_markers(progress_dir: Path) -> set[str]:
    """Names of every ``P<k>.retry*`` marker currently in the progress dir."""
    return {p.name for p in progress_dir.glob("P*.retry*")}


_RETRY_PHASE = re.compile(r"^(P\d+)\.retry")


def newly_reopened(current_id: str, before: set[str], after: set[str]) -> Optional[str]:
    """The earliest phase that a session of ``current_id`` re-opened by
    writing a new retry marker for a phase that runs *before* it; None if
    the session re-opened nothing. A verdict phase that sends an earlier
    phase back is then finished without its own done marker."""
    current_index = PHASES.index(phase_by_id(current_id))
    reopened: list[int] = []
    for name in after - before:
        match = _RETRY_PHASE.match(name)
        if not match:
            continue
        index = PHASES.index(phase_by_id(match.group(1)))
        if index < current_index:
            reopened.append(index)
    if not reopened:
        return None
    return PHASES[min(reopened)].id


_COMMIT_LINE = re.compile(r"^commit:\s*([0-9a-f]{7,40})\s*$")


def validate_done_marker(
    progress_dir: Path,
    phase_id: str,
    *,
    head_sha: str,
    porcelain: str,
    work_root: Path,
) -> list[str]:
    """Every way the phase's deliverables fall short; empty means accepted."""
    phase = phase_by_id(phase_id)
    problems: list[str] = []
    marker = progress_dir / f"{phase_id}.done"
    if not marker.is_file():
        problems.append(f"{marker.name} is missing from {progress_dir}")
    else:
        first = (marker.read_text(encoding="utf-8").splitlines() or [""])[0]
        match = _COMMIT_LINE.match(first)
        if not match:
            problems.append(
                f"{marker.name}: the first line must be `commit: <sha of HEAD>`; got {first!r}"
            )
        elif not head_sha.startswith(match.group(1)):
            problems.append(
                f"{marker.name} names commit {match.group(1)} but HEAD is {head_sha}"
            )
    if porcelain.strip():
        shown = "\n".join(porcelain.strip().splitlines()[:10])
        problems.append(
            "uncommitted changes in the clone (commit them on the branch or delete them):\n"
            + shown
        )
    if phase_id == "P0":
        for name in (BASELINE_FAILING_NAME, BASELINE_COLLECTION_NAME):
            if not (progress_dir / name).is_file():
                problems.append(f"P0 must write {progress_dir / name}")
    if phase.jobs_file:
        try:
            parse_jobs_file(progress_dir / phase.jobs_file, phase.required_labels)
        except ValueError as exc:
            problems.append(str(exc))
    for name in phase.requires_files:
        if not (work_root / name).is_file():
            problems.append(f"{phase_id} must write {work_root / name}")
    return problems


# --- jobs files ---------------------------------------------------------------------


def parse_jobs_file(path: Path, required_labels: Sequence[str]) -> list[str]:
    """Every Slurm job id in a jobs file, in file order. The file maps a label
    (an arm, or ``analysis``) to ``{"work_root": ..., "job_ids": [...]}``.
    Raises ``ValueError`` on a missing file, bad JSON, a missing required
    label, an empty id list, or a non-numeric id."""
    if not path.is_file():
        raise ValueError(f"{path.name} is missing from {path.parent}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an object keyed by label")
    for label in required_labels:
        if label not in data:
            raise ValueError(f"{path}: required label {label!r} is absent")
    ids: list[str] = []
    for label, entry in data.items():
        if not isinstance(entry, dict) or not entry.get("work_root"):
            raise ValueError(f"{path}: label {label!r} must be an object with work_root and job_ids")
        job_ids = entry.get("job_ids")
        if not isinstance(job_ids, list) or not job_ids:
            raise ValueError(f"{path}: label {label!r} has no job_ids")
        for job_id in job_ids:
            if not str(job_id).isdigit():
                raise ValueError(f"{path}: label {label!r} has a non-numeric job id {job_id!r}")
            ids.append(str(job_id))
    return ids


_ARRAY_ID = re.compile(r"^(\d+)")


def jobs_still_queued(squeue_output: str) -> set[str]:
    """Base job ids present in ``squeue -h -o %i`` output (array tasks such as
    ``123_4`` or ``123_[2-5%2]`` collapse to ``123``)."""
    ids: set[str] = set()
    for line in squeue_output.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _ARRAY_ID.match(line)
        if match:
            ids.add(match.group(1))
    return ids


# --- prompt, env, requeue ------------------------------------------------------------


def compose_prompt(template: str, mapping: Mapping[str, str]) -> str:
    """``$placeholder`` substitution; an unknown placeholder raises."""
    try:
        return string.Template(template).substitute(mapping)
    except KeyError as exc:
        raise ValueError(f"prompt template uses an unknown placeholder: {exc}") from exc


def load_env(path: Path) -> dict[str, str]:
    """``KEY=value`` lines (``#`` comments and blanks ignored; surrounding
    quotes stripped). Raises if the file is missing."""
    if not path.is_file():
        raise ValueError(f"env file not found: {path}")
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}: not a KEY=value line: {raw!r}")
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def requeue_command(
    sbatch_path: Path,
    *,
    job_name: str,
    partition: str,
    time: str,
    cpus: str,
    mem: str,
    log_dir: Path,
    export: str,
    dependency: Optional[str] = None,
    begin: Optional[str] = None,
) -> list[str]:
    """argv that resubmits the consolidation job (same script, same env), to
    start after ``dependency`` jobs and/or at ``begin``."""
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--partition={partition}",
        f"--time={time}",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--output={log_dir}/%x_%j.out",
        f"--error={log_dir}/%x_%j.out",
        "--mail-type=END,FAIL,TIMEOUT",
        f"--export={export}",
    ]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    if begin:
        cmd.append(f"--begin={begin}")
    cmd.append(str(sbatch_path))
    return cmd


# --- walltime ------------------------------------------------------------------------------


def seconds_left(text: str) -> Optional[int]:
    """Parse ``squeue -o %L`` (``D-HH:MM:SS``, ``HH:MM:SS``, ``MM:SS``, ``SS``).
    ``UNLIMITED`` / ``NOT_SET`` / ``INVALID`` give None; anything else raises."""
    text = text.strip()
    if text.upper() in ("UNLIMITED", "NOT_SET", "INVALID", "N/A"):
        return None
    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        if not day_part.isdigit():
            raise ValueError(f"unparseable squeue time-left value: {text!r}")
        days = int(day_part)
    parts = text.split(":")
    if not parts or not all(p.isdigit() for p in parts) or len(parts) > 3:
        raise ValueError(f"unparseable squeue time-left value: {text!r}")
    numbers = [int(p) for p in parts]
    while len(numbers) < 3:
        numbers.insert(0, 0)
    hours, minutes, seconds = numbers
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def needs_walltime_requeue(
    *, seconds_left: Optional[int], session_timeout: int, margin: int
) -> bool:
    """True when a full agent session (plus the driver's margin) would not
    fit in what remains of this job's walltime."""
    if seconds_left is None:
        return False
    return seconds_left < session_timeout + margin


def render_inputs(env: Mapping[str, str], keys: Sequence[str]) -> str:
    """The frozen-inputs block for the prompt, ``KEY=value`` per line."""
    missing = [k for k in keys if k not in env]
    if missing:
        raise ValueError(f"consolidation.env lacks {missing}")
    return "\n".join(f"{k}={env[k]}" for k in keys)
