"""Pure parts of the consolidation driver.

The consolidation plan (``docs/consolidation_plan_2026_09.md``) is executed as
nine phases, P0..P8, one Claude Code session each. State lives on disk under
``<work_root>/progress/`` as marker files the agent writes and the driver
validates:

* ``P<k>.done`` — first line ``commit: <sha>`` (must equal HEAD), then a summary;
* ``P<k>.blocked`` — the agent hit a stop condition; the job stops;
* ``smoke_jobs.json`` (P7) — the Slurm ids the driver must wait on before P8;
* ``P7.retry*`` — P8 asked for another smoke round (bounded).

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
SMOKE_JOBS_NAME = "smoke_jobs.json"
BASELINE_FAILING_NAME = "baseline_failing_tests.txt"
BASELINE_COLLECTION_NAME = "baseline_collection_errors.txt"
VERDICT_NAME = "VERDICT.md"
HANDOFF_NAME = "HANDOFF.md"
MAX_SMOKE_ROUNDS = 2

# Tools the agent must never use, enforced by Claude Code on top of the plan's
# rules. ``sbatch`` is added for every phase but the smoke submission (P7).
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
    waits_for_smoke: bool = False


PHASES: tuple[Phase, ...] = (
    Phase("P0", "Baseline on the untouched base"),
    Phase("P1", "Integrate: merge main, restore the audit, merge arm C"),
    Phase("P2", "Make raw mode genuinely raw end to end"),
    Phase("P3", "Honest metrics and offline diagnostics"),
    Phase("P4", "Reimplement the race presentation"),
    Phase("P5", "Reimplement lens rotation"),
    Phase("P6", "Documentation, decision record, full checks"),
    Phase("P7", "Submit the two smoke chains", allows_sbatch=True),
    Phase("P8", "Smoke verdict and handoff", waits_for_smoke=True),
)


class Blocked(RuntimeError):
    """A phase wrote its ``.blocked`` marker: the job must stop, not continue."""


def phase_by_id(phase_id: str) -> Phase:
    for phase in PHASES:
        if phase.id == phase_id:
            return phase
    raise KeyError(f"no consolidation phase {phase_id!r}; known: {[p.id for p in PHASES]}")


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
    if phase_id == "P7":
        try:
            parse_smoke_jobs(progress_dir / SMOKE_JOBS_NAME)
        except ValueError as exc:
            problems.append(str(exc))
    if phase_id == "P8":
        for name in (VERDICT_NAME, HANDOFF_NAME):
            if not (work_root / name).is_file():
                problems.append(f"P8 must write {work_root / name}")
    return problems


# --- smoke jobs -------------------------------------------------------------------


def parse_smoke_jobs(path: Path) -> list[str]:
    """Every Slurm job id the P8 phase must wait for, in file order.
    Raises ``ValueError`` on a missing file, a missing arm, or a non-numeric id."""
    if not path.is_file():
        raise ValueError(f"{path.name} is missing from {path.parent}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    ids: list[str] = []
    for arm in ("featurized", "raw"):
        entry = data.get(arm)
        if not isinstance(entry, dict) or not entry.get("work_root"):
            raise ValueError(f"{path}: arm {arm!r} must be an object with work_root and job_ids")
        job_ids = entry.get("job_ids")
        if not isinstance(job_ids, list) or not job_ids:
            raise ValueError(f"{path}: arm {arm!r} has no job_ids")
        for job_id in job_ids:
            if not str(job_id).isdigit():
                raise ValueError(f"{path}: arm {arm!r} has a non-numeric job id {job_id!r}")
            ids.append(str(job_id))
    return ids


def smoke_round(progress_dir: Path) -> int:
    """1 for the first smoke submission; +1 per ``P7.retry*`` marker P8 wrote."""
    return 1 + len(list(progress_dir.glob("P7.retry*")))


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


# --- prompt, env, requeue -------------------------------------------------------------


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
