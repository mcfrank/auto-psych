"""Submitting a step's fits and its continuation (`DISTRIBUTED_DESIGN.md` v3, §0a-§0b).

When a step's phase has fits pending (`src.rsa_open.loop.fit_array.FitsPending`), `submit_step`:
1. refuses, writing STOP so that it mails, if the cell is stopped, this launch's step budget is
   spent, or the user already has too many jobs queued (one ``squeue`` count, never a polling loop);
2. writes one manifest per memory size and submits each as a **held** array of single-core fits
   (`scripts/rsa/slurm/fit_array.sbatch`), throttled, on the fit partitions and CPU type, with a
   clean environment (``--export=NONE`` and only what a fit needs: never ``.secrets``, since a fit
   runs agent-written code);
   then the folds to fit after them (`FoldAfter`), held too, starting when those arrays end;
3. submits the chain's continuation (`continuation.sh`, which the step's own job writes) after
   all of them (``afterany``) and after the step itself (``singleton`` on the chain's job name, so
   one step per chain runs at a time);
4. releases the arrays, and records the step in ``steps.jsonl``.
If any call fails, whatever was submitted is cancelled and the cell gets a STOP: a step never
leaves a chain with fits but no continuation, or the reverse.

**A launch** is what a person starts (SRCC ticket #106983, 2026-11-05: a person starts one job per
experiment, and the job's successors run that experiment's phases, with a low, discrete cap). Its
first step records `launch.json` (`begin_launch`): the experiment it runs and the steps before it.
The step budget, `MAX_STEPS`, is per launch; the chain stops after that experiment
(`src.rsa_open.chain.ExperimentDone`), and a person launches the next.
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

from src.rsa_open.loop.fit_array import FitsPending, write_after_manifest, write_manifest

MAX_STEPS = 40
"""Steps per launch (one experiment) before a person is asked: the Sherlock rehearsal took 2-3 a
round and 3 more for the starting fits and the end, about 18 for 5 rounds (review V3 §11)."""
MAX_QUEUED = 1500
"""The user's queued and running jobs (array tasks counted) above which no step submits more."""
THROTTLE = 24
TIME_MARGIN_SEC = 20 * 60
NO_LIMIT_TIME = "2-00:00:00"

Runner = Callable[[Sequence[str]], str]


class StepRefused(RuntimeError):
    """A step that would not submit (stopped, its budget spent, or the queue full); its STOP is written."""


class StepYield(Exception):
    """The step has submitted its continuation and ends here (`EXIT_YIELD`): no STOP, no mail."""

    def __init__(self, record: Mapping[str, object]) -> None:
        super().__init__(f"continued in job {record.get('continuation')}")
        self.record = record


EXIT_YIELD = 6
"""The step's exit code when it handed over to its continuation (open_live.sbatch: exit 0, no STOP)."""
EXIT_REFUSED = 7
"""The step's exit code when it refused to submit (its STOP is written; open_live.sbatch mails)."""


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
                   None if env.get("RSA_OPEN_FIT_CONSTRAINT", "CPU_GEN:SIE") in ("", "none")
                   else env.get("RSA_OPEN_FIT_CONSTRAINT", "CPU_GEN:SIE"))


def write_stop(cell_dir: Path, reason: str) -> None:
    (Path(cell_dir) / "STOP").write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\t{reason}\n")


def launch_path(cfg: StepConfig) -> Path:
    return cfg.cell_dir / "launch.json"


def begin_launch(cfg: StepConfig, experiment: int, max_steps: Optional[int] = None) -> dict:
    """Record a person's launch: the experiment it runs and the steps before it (its budget counts from here)."""
    record = {"experiment": int(experiment), "steps_before": len(_steps(cfg)),
              "max_steps": int(max_steps if max_steps is not None else cfg.max_steps), "at": time.time(),
              "job": os.environ.get("SLURM_JOB_ID")}
    tmp = launch_path(cfg).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, launch_path(cfg))
    with (cfg.cell_dir / "launches.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def current_launch(cfg: StepConfig) -> Optional[dict]:
    path = launch_path(cfg)
    return json.loads(path.read_text()) if path.exists() else None


def _steps(cfg: StepConfig) -> List[dict]:
    """The chain's steps that submitted a continuation (an early array alone isn't a step)."""
    if not cfg.steps_log.exists():
        return []
    entries = [json.loads(line) for line in cfg.steps_log.read_text().splitlines() if line.strip()]
    return [e for e in entries if e.get("continuation")]


def queued_jobs(runner: Runner) -> int:
    """The user's jobs, each array task counted (one ``squeue`` call)."""
    return sum(1 for line in runner(["squeue", "--me", "-h", "-r", "-o", "%i"]).splitlines() if line.strip())


def _time_arg(limits: Sequence[float]) -> str:
    finite = [t for t in limits if t is not None and math.isfinite(t)]
    if len(finite) < len(limits):
        return NO_LIMIT_TIME
    return str(int(math.ceil((max(finite) + TIME_MARGIN_SEC) / 60)))  # minutes


def _check(cfg: StepConfig, tasks: int, runner: Runner, *, step: bool) -> int:
    if (cfg.cell_dir / "STOP").exists():
        raise StepRefused(f"{cfg.cell_dir.name} is stopped: no step submitted")
    done = _steps(cfg)
    launch = current_launch(cfg)
    used = len(done) - (launch["steps_before"] if launch else 0)
    cap = launch["max_steps"] if launch else cfg.max_steps
    if step and used >= cap:
        why = (f"step budget spent: {used} steps submitted " + (f"in experiment {launch['experiment']}'s launch "
               if launch else "") + f"(at most {cap}); a person looks")
        write_stop(cfg.cell_dir, why)
        raise StepRefused(why)
    queued = queued_jobs(runner)
    if queued + tasks > cfg.max_queued:
        why = f"{queued} jobs already queued and {tasks} fits to submit: over {cfg.max_queued}; a person looks"
        write_stop(cfg.cell_dir, why)
        raise StepRefused(why)
    return len(done)


def _arrays(cfg: StepConfig, pending: Sequence, runner: Runner, stamp: str, *, hold: bool) -> List[str]:
    """Submit ``pending`` as one array per memory size; returns the array job ids."""
    by_size: Dict[int, list] = defaultdict(list)
    for p in pending:
        by_size[p.mem_gb].append(p)
    export = "NONE," + ",".join(f"{k}={v}" for k, v in cfg.fit_env.items())
    ids: List[str] = []
    for gb, group in sorted(by_size.items()):
        manifest = write_manifest(cfg.arrays_dir / f"{stamp}.mem{gb}.json", group)
        cmd = ["sbatch", "--parsable"] + (["--hold"] if hold else []) + [
               f"--array=0-{len(group) - 1}%{cfg.throttle}", f"--mem={gb}G",
               "--cpus-per-task=1", f"--time={_time_arg([p.job.time_limit_sec for p in group])}",
               f"--partition={cfg.fit_partition}", f"--job-name={cfg.chain_job}_fits",
               f"--output={cfg.logs_dir}/%x_%A_%a.out", f"--export={export}"]
        if cfg.fit_constraint:
            cmd.append(f"--constraint={cfg.fit_constraint}")
        ids.append(runner(cmd + [str(cfg.fit_script), str(manifest)]).strip().split(";")[0])
    return ids


def _after_arrays(cfg: StepConfig, after: Sequence, runner: Runner, stamp: str, *, hold: bool,
                  depend: Sequence[str]) -> List[str]:
    """Submit folds to fit after their full fits (one array per memory size), after ``depend``."""
    by_size: Dict[int, list] = defaultdict(list)
    for a in after:
        by_size[a.mem_gb].append(a)
    export = "NONE," + ",".join(f"{k}={v}" for k, v in cfg.fit_env.items())
    ids: List[str] = []
    for gb, group in sorted(by_size.items()):
        manifest = write_after_manifest(cfg.arrays_dir / f"{stamp}.after.mem{gb}.json", group)
        cmd = ["sbatch", "--parsable"] + (["--hold"] if hold else []) + (
               [f"--dependency=afterany:{':'.join(depend)}"] if depend else []) + [
               f"--array=0-{len(group) - 1}%{cfg.throttle}", f"--mem={gb}G",
               "--cpus-per-task=1", f"--time={NO_LIMIT_TIME}",
               f"--partition={cfg.fit_partition}", f"--job-name={cfg.chain_job}_fits",
               f"--output={cfg.logs_dir}/%x_%A_%a.out", f"--export={export}"]
        if cfg.fit_constraint:
            cmd.append(f"--constraint={cfg.fit_constraint}")
        ids.append(runner(cmd + [str(cfg.fit_script), str(manifest)]).strip().split(";")[0])
    return ids


def _sizes(pending: Sequence) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for p in pending:
        out[str(p.mem_gb)] += 1
    return dict(out)


def _log(cfg: StepConfig, record: Mapping[str, object]) -> None:
    cfg.steps_log.parent.mkdir(parents=True, exist_ok=True)
    with cfg.steps_log.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _cancel_and_stop(cfg: StepConfig, runner: Runner, ids: Sequence[str], exc: BaseException) -> None:
    for jid in ids:
        try:
            runner(["scancel", jid])
        except Exception:  # noqa: BLE001 (cancelling what we can; the STOP below says what happened)
            pass
    write_stop(cfg.cell_dir, f"a step's submission failed ({type(exc).__name__}: {exc}); "
                             "anything submitted was cancelled; a person looks")


def submit_early(cfg: StepConfig, pending: FitsPending, *, runner: Runner = run_command,
                 reason: str = "", depend_on: Sequence[str] = ()) -> List[str]:
    """Submit fits a later step will need, now, without a continuation (an admission's folds, while
    the repair agents work); the step that yields depends on them. Returns the array ids."""
    stamp = time.strftime("%Y%m%dT%H%M%S") + "_early"
    _check(cfg, len(pending.pending) + len(pending.after), runner, step=False)
    ids: List[str] = []
    try:
        ids = _arrays(cfg, pending.pending, runner, stamp, hold=False)
        if pending.after:
            ids += _after_arrays(cfg, pending.after, runner, stamp, hold=False, depend=list(ids) + list(depend_on))
    except Exception as exc:
        _cancel_and_stop(cfg, runner, ids, exc)
        raise RuntimeError(f"submitting early fits failed: {exc}") from exc
    _log(cfg, {"at": time.time(), "reason": reason, "early": True, "arrays": ids, "tasks": _sizes(pending.pending),
               "folds_after": len(pending.after)})
    return ids


def submit_step(cfg: StepConfig, pending: Optional[FitsPending], *, runner: Runner = run_command,
                reason: str = "", depend_on: Sequence[str] = ()) -> Dict[str, object]:
    """Submit ``pending`` as held arrays, the continuation after them (and after ``depend_on``, arrays
    submitted early by this step), then release them; returns the step's record. Raises `StepRefused`
    (STOP written) or, on a failed call, RuntimeError (STOP written, anything submitted cancelled)."""
    items = list(pending.pending) if pending is not None else []
    after = list(pending.after) if pending is not None else []
    if not items and not after and not depend_on:
        raise ValueError("a step yields only to wait for fits")
    done = _check(cfg, len(items) + len(after), runner, step=True)
    stamp = time.strftime("%Y%m%dT%H%M%S") + f"_{done + 1}"
    arrays: List[str] = []
    continuation = None
    try:
        arrays = _arrays(cfg, items, runner, stamp, hold=True)
        if after:
            arrays += _after_arrays(cfg, after, runner, stamp, hold=True,
                                    depend=list(arrays) + [d for d in depend_on if d not in arrays])
        after = list(arrays) + [d for d in depend_on if d not in arrays]
        continuation = runner(["bash", str(cfg.continuation), f"afterany:{':'.join(after)},singleton"]).strip()
        if not continuation:
            raise RuntimeError("the continuation script printed no job id")
        if arrays:
            runner(["scontrol", "release", ",".join(arrays)])
    except Exception as exc:
        _cancel_and_stop(cfg, runner, arrays + ([continuation] if continuation else []), exc)
        raise RuntimeError(f"submitting the step failed: {exc}") from exc
    record = {"at": time.time(), "reason": reason, "arrays": arrays, "after_early": list(depend_on),
              "continuation": continuation, "tasks": _sizes(items), "folds_after": len(after)}
    _log(cfg, record)
    return record


class Stepper:
    """This step's submissions: fits submitted early (by fit and memory size) and the final yield."""

    def __init__(self, cfg: StepConfig, runner: Runner = run_command) -> None:
        self.cfg, self.runner = cfg, runner
        self.inflight: Dict[str, str] = {}

    @staticmethod
    def _key(p) -> str:
        return f"{p.job.nc_path}|{p.mem_gb}"

    def submit_early(self, pending: FitsPending, reason: str = "") -> None:
        new = [p for p in pending.pending if self._key(p) not in self.inflight]
        later = [a for a in pending.after if a.key not in self.inflight]
        if not new and not later:
            return
        ids = submit_early(self.cfg, FitsPending(new, later), runner=self.runner, reason=reason,
                           depend_on=sorted(set(self.inflight.values())) if later else ())
        sizes = sorted({p.mem_gb for p in new})
        by_size = {gb: jid for gb, jid in zip(sizes, ids)}
        for p in new:
            self.inflight[self._key(p)] = by_size[p.mem_gb]
        later_ids = ids[len(sizes):]
        for a in later:
            self.inflight[a.key] = dict(zip(sorted({x.mem_gb for x in later}), later_ids))[a.mem_gb]

    def yield_step(self, pending: FitsPending, reason: str = "") -> None:
        """Submit what isn't already running, the continuation after everything, and end the step."""
        new = [p for p in pending.pending if self._key(p) not in self.inflight]
        later = [a for a in pending.after if a.key not in self.inflight]
        # Every array this step submitted early, needed now or not: a continuation that started while
        # one ran would find its fits pending and submit them a second time.
        early = sorted(set(self.inflight.values()))
        record = submit_step(self.cfg, FitsPending(new, later) if new or later else None, runner=self.runner,
                             reason=reason, depend_on=early)
        raise StepYield(record)


_ACTIVE: Optional[Stepper] = None


def activate(stepper: Optional[Stepper]) -> None:
    """Step mode for this process (run_chain sets it from the job's environment)."""
    global _ACTIVE
    _ACTIVE = stepper


def active() -> Optional[Stepper]:
    return _ACTIVE
