"""Submitting a step's fits and its continuation (`DISTRIBUTED_DESIGN.md` v3, §0a-§0b).

When a step's phase has fits pending (`src.rsa_open.loop.fit_array.FitsPending`), `submit_step`:
1. refuses, writing STOP so that it mails, if the cell is stopped, the step budget is spent, or
   the user already has too many jobs queued (one ``squeue`` count, never a polling loop);
2. writes one manifest per memory size and submits each as a **held** array of single-core fits
   (`scripts/rsa/slurm/fit_array.sbatch`), throttled, on the fit partitions and CPU type, with a
   clean environment (``--export=NONE`` and only what a fit needs: never ``.secrets``, since a fit
   runs agent-written code);
3. submits the chain's continuation (`continuation.sh`, which the step's own job writes) after
   all of them (``afterany``) and after the step itself (``singleton`` on the chain's job name, so
   one step per chain runs at a time);
4. releases the arrays, and records the step in ``steps.jsonl``.
If any call fails, whatever was submitted is cancelled and the cell gets a STOP: a step never
leaves a chain with fits but no continuation, or the reverse.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from src.rsa_open.loop.fit_array import FitsPending, write_manifest

MAX_STEPS = 80
"""Steps per chain before a person is asked (about 4 a round, 5 rounds, 3-4 experiments, and room)."""
MAX_QUEUED = 1500
"""The user's queued and running jobs (array tasks counted) above which no step submits more."""
THROTTLE = 24
TIME_MARGIN_SEC = 20 * 60
NO_LIMIT_TIME = "2-00:00:00"

Runner = Callable[[Sequence[str]], str]


class StepRefused(RuntimeError):
    """A step that would not submit (stopped, its budget spent, or the queue full); its STOP is written."""


def run_command(cmd: Sequence[str]) -> str:
    return subprocess.run(list(cmd), capture_output=True, text=True, check=True, timeout=300).stdout


@dataclass(frozen=True)
class StepConfig:
    cell_dir: Path
    chain_job: str
    """The chain's job name: its steps share it, so ``singleton`` runs one at a time."""
    continuation: Path
    """``continuation.sh <dependency>``: submits the chain's next step and prints its job id."""
    fit_script: Path
    fit_env: Mapping[str, str]
    logs_dir: Path
    fit_partition: str = "mcfrank,owners"
    fit_constraint: Optional[str] = "CPU_GEN:SIE"
    throttle: int = THROTTLE
    max_steps: int = MAX_STEPS
    max_queued: int = MAX_QUEUED

    @property
    def arrays_dir(self) -> Path:
        return self.cell_dir / "arrays"

    @property
    def steps_log(self) -> Path:
        return self.cell_dir / "steps.jsonl"

    @classmethod
    def from_env(cls) -> "StepConfig":
        """The step's settings, as `scripts/rsa/slurm/open_live.sbatch` exports them in step mode."""
        env = os.environ
        need = ["RSA_OPEN_CELL_DIR", "RSA_OPEN_CHAIN_JOB", "RSA_OPEN_CONTINUATION", "RSA_OPEN_FIT_SCRIPT",
                "RSA_OPEN_LOGS", "VENV_PY", "HARNESS_REPO", "JAX_COMPILATION_CACHE_DIR"]
        missing = [k for k in need if not env.get(k)]
        if missing:
            raise RuntimeError(f"step mode needs {missing} in the environment (scripts/rsa/slurm/open_live.sbatch)")
        fit_env = {k: env[k] for k in ("VENV_PY", "HARNESS_REPO", "JAX_COMPILATION_CACHE_DIR")}
        return cls(Path(env["RSA_OPEN_CELL_DIR"]), env["RSA_OPEN_CHAIN_JOB"], Path(env["RSA_OPEN_CONTINUATION"]),
                   Path(env["RSA_OPEN_FIT_SCRIPT"]), fit_env, Path(env["RSA_OPEN_LOGS"]),
                   env.get("RSA_OPEN_FIT_PARTITION", "mcfrank,owners"),
                   env.get("RSA_OPEN_FIT_CONSTRAINT", "CPU_GEN:SIE") or None)


def write_stop(cell_dir: Path, reason: str) -> None:
    (Path(cell_dir) / "STOP").write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\t{reason}\n")


def _steps(cfg: StepConfig) -> List[dict]:
    if not cfg.steps_log.exists():
        return []
    return [json.loads(line) for line in cfg.steps_log.read_text().splitlines() if line.strip()]


def queued_jobs(runner: Runner) -> int:
    """The user's jobs, each array task counted (one ``squeue`` call)."""
    return sum(1 for line in runner(["squeue", "--me", "-h", "-r", "-o", "%i"]).splitlines() if line.strip())


def _time_arg(limits: Sequence[float]) -> str:
    finite = [t for t in limits if t is not None and math.isfinite(t)]
    if len(finite) < len(limits):
        return NO_LIMIT_TIME
    return str(int(math.ceil((max(finite) + TIME_MARGIN_SEC) / 60)))  # minutes


def submit_step(cfg: StepConfig, pending: FitsPending, *, runner: Runner = run_command,
                reason: str = "") -> Dict[str, object]:
    """Submit ``pending`` as held arrays, the continuation after them, then release them; returns the
    step's record. Raises `StepRefused` (STOP written) or, on a failed call, RuntimeError (STOP
    written, anything submitted cancelled)."""
    if (cfg.cell_dir / "STOP").exists():
        raise StepRefused(f"{cfg.cell_dir.name} is stopped: no step submitted")
    done = _steps(cfg)
    if len(done) >= cfg.max_steps:
        why = f"step budget spent: {len(done)} steps submitted (at most {cfg.max_steps}); a person looks"
        write_stop(cfg.cell_dir, why)
        raise StepRefused(why)
    tasks = len(pending.pending)
    queued = queued_jobs(runner)
    if queued + tasks > cfg.max_queued:
        why = f"{queued} jobs already queued and {tasks} fits to submit: over {cfg.max_queued}; a person looks"
        write_stop(cfg.cell_dir, why)
        raise StepRefused(why)
    by_size: Dict[int, list] = defaultdict(list)
    for p in pending.pending:
        by_size[p.mem_gb].append(p)
    stamp = time.strftime("%Y%m%dT%H%M%S") + f"_{len(done) + 1}"
    export = "NONE," + ",".join(f"{k}={v}" for k, v in cfg.fit_env.items())
    arrays: List[str] = []
    continuation = None
    try:
        for gb, group in sorted(by_size.items()):
            manifest = write_manifest(cfg.arrays_dir / f"{stamp}.mem{gb}.json", group)
            cmd = ["sbatch", "--parsable", "--hold", f"--array=0-{len(group) - 1}%{cfg.throttle}", f"--mem={gb}G",
                   "--cpus-per-task=1", f"--time={_time_arg([p.job.time_limit_sec for p in group])}",
                   f"--partition={cfg.fit_partition}", f"--job-name={cfg.chain_job}_fits",
                   f"--output={cfg.logs_dir}/%x_%A_%a.out", f"--export={export}"]
            if cfg.fit_constraint:
                cmd.append(f"--constraint={cfg.fit_constraint}")
            arrays.append(runner(cmd + [str(cfg.fit_script), str(manifest)]).strip().split(";")[0])
        continuation = runner(["bash", str(cfg.continuation), f"afterany:{':'.join(arrays)},singleton"]).strip()
        if not continuation:
            raise RuntimeError("the continuation script printed no job id")
        runner(["scontrol", "release", ",".join(arrays)])
    except Exception as exc:
        for jid in arrays + ([continuation] if continuation else []):
            try:
                runner(["scancel", jid])
            except Exception:  # noqa: BLE001 (cancelling what we can; the STOP below says what happened)
                pass
        write_stop(cfg.cell_dir, f"a step's submission failed ({type(exc).__name__}: {exc}); "
                                 "anything submitted was cancelled; a person looks")
        raise RuntimeError(f"submitting the step failed: {exc}") from exc
    record = {"at": time.time(), "reason": reason, "arrays": arrays, "continuation": continuation,
              "tasks": {str(gb): len(g) for gb, g in by_size.items()}}
    cfg.steps_log.parent.mkdir(parents=True, exist_ok=True)
    with cfg.steps_log.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record
