"""The designer's tools (docs/auto_open/PLAN.md §4): a CLI the designer agent runs in its sandbox.

    <python> -m src.rsa_open.tools validate   --workspace <ws> --spec <ws>/work/p.json
    <python> -m src.rsa_open.tools eig        --workspace <ws> --spec <ws>/work/p.json
    <python> -m src.rsa_open.tools link-check --workspace <ws> --spec <ws>/work/p.json
    <python> -m src.rsa_open.tools submit     --workspace <ws> --spec <ws>/work/p.json

**The workspace** (`Workspace`) is one designer session's directory, laid out by the harness
(`src.rsa_open.designer`):
- ``context/``: the design context (`src.rsa_open.design`), with the closed paradigm's reference
  score already computed; read-only to the agent;
- ``inputs/``: the brief, the spec's JSON schema, the link library, the chain's earlier
  paradigms and the timing records (``timing.json``, `src.rsa_open.timing`); read-only;
- ``work/``: the agent's own directory (its specs, ``reports/``, ``tool_log.jsonl`` and
  ``submission.json``).

**The commands:**
- ``validate``: the spec's problems, all at once (`src.rsa_open.paradigm`), the trial units
  against the EIG tool's limit, and the pilot's time estimate against the declared minutes, on a
  random design from the scene space;
- ``eig``: the paradigm's design under the live theories, its bits per dollar and the floor, and
  the time estimate on that design; ``--quick`` scores it with ``QUICK_DRAWS`` draws and
  ``QUICK_SCENARIOS`` scenarios, for exploring (saved as ``eig-quick_<sha12>.json``, never
  read by ``submit``);
- ``link-check``: the recovery test (`src.rsa_open.fit.recovery_check`) of the spec's link **in
  this paradigm**: its scene space, its number of responses, its scale, simulated from a live
  theory at its posterior mean and the link at its reference values. The theory is the one with
  the most parameters unless named (`default_theory`): under the literal listener, which is flat
  over the true objects, a listener link's temperature changes nothing, so no paradigm could
  recover it (the 2026-10-21 smoke failed every choice paradigm that way);
- ``submit``: records the spec as the session's submission, once it has an ``eig`` report that
  meets the floor and a ``link-check`` report that recovers, both of this exact spec (sha256).

Every report is written to ``work/reports/<command>_<sha12>.json``. ``submit`` only records:
the harness then re-scores the spec with a seed of its own, reruns the link check, and pilots
it (`src.rsa_open.designer`); the agent's reports are never trusted as the gate.

**The clock.** Before each session the harness writes ``inputs/deadline.json`` (when the
session is killed); every command's output ends with the time left (`time_note`), and with an
urging note in the last ``URGENT_SECONDS``. A session that ends without a submission is
submitted for by the harness: its best spec with a full ``eig`` report that meets the floor
(`best_scored_spec`), which then goes through every check a submission does. Run B's
experiment-3 designer spent 92 of its 120 minutes on its own scripts before its first tool call,
had a full eig of a spec well above the floor at minute 119, and submitted nothing (2026-10-24).

**The call budget.** Every command is a line in ``work/tool_log.jsonl``, and a command past
``MAX_TOOL_CALLS`` is refused (PI 2026-10-21: 40 calls). The log is in the agent's own
directory; the harness counts it too and the agent's transcript records every command.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, Union

import numpy as np
import tyro

MAX_TOOL_CALLS = 40
TOOL_LOG = "tool_log.jsonl"
SUBMISSION = "submission.json"
DEADLINE = "deadline.json"
URGENT_SECONDS = 30 * 60
# Link check: NUTS settings and the replicates of the library's recovery test.
LINK_CHECK_WARMUP = 500
LINK_CHECK_SAMPLES = 500
# eig --quick: about a twentieth of the full score's draws x scenarios (2026-10-21: a full eig took
# ~6.5 min of the first designer's 114, its greedy selection most of that).
QUICK_DRAWS = 16
QUICK_SCENARIOS = 200


class ToolRefused(RuntimeError):
    """A command the tools will not run (budget spent, workspace broken, prerequisites missing)."""


@dataclass(frozen=True)
class Workspace:
    root: Path

    @property
    def context(self) -> Path:
        return self.root / "context"

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def reports(self) -> Path:
        return self.work / "reports"

    @property
    def log(self) -> Path:
        return self.work / TOOL_LOG

    @property
    def submission(self) -> Path:
        return self.work / SUBMISSION

    @property
    def deadline(self) -> Path:
        return self.inputs / DEADLINE

    @property
    def timing(self) -> Path:
        return self.inputs / "timing.json"

    def timing_records(self) -> list:
        from src.rsa_open.timing import read_records

        if not self.timing.exists():
            raise ToolRefused(f"{self.root}: no inputs/timing.json (the harness writes it)")
        return read_records(self.timing)

    def check(self) -> None:
        missing = [p.name for p in (self.context, self.inputs, self.work) if not p.is_dir()]
        if missing:
            raise ToolRefused(f"{self.root} is not a designer workspace: no {missing}")

    def report_path(self, command: str, sha: str) -> Path:
        return self.reports / f"{command}_{sha[:12]}.json"

    def calls(self) -> list:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]


def write_deadline(ws: Workspace, deadline: float) -> None:
    """The session's end (epoch seconds), for `time_note` (the harness writes it; the agent may not)."""
    ws.deadline.write_text(json.dumps({"deadline": deadline, "clock": time.strftime("%H:%M %Z", time.localtime(deadline))}))


def time_note(ws: Workspace, now: Optional[float] = None) -> Optional[str]:
    """How long the session has left, for every command's output (None without a deadline)."""
    if not ws.deadline.exists():
        return None
    d = json.loads(ws.deadline.read_text())
    left = d["deadline"] - (time.time() if now is None else now)
    note = f"time: {max(0, left) / 60:.0f} min left in this session (it ends at {d['clock']})"
    if left < URGENT_SECONDS:
        note += (". SUBMIT NOW: run the full eig on the spec you mean to submit, then link-check and submit. "
                 "If the session ends without a submission, the harness submits your best spec that has a full "
                 "eig meeting the floor")
    return note


def best_scored_spec(ws: Workspace, exclude=frozenset()) -> Optional[Dict[str, Any]]:
    """The spec in ``work/`` that a session ending without a submission is submitted with: valid
    as it stands, with a full ``eig`` report of that exact spec that meets the floor, the highest
    bits per dollar; None if there is none. ``exclude``: sha256s that already failed."""
    from src.rsa_open.paradigm import ParadigmInvalid, load_designed

    best = None
    for spec in sorted(ws.work.glob("*.json")):
        if spec.name in (SUBMISSION,):
            continue
        try:
            p = load_designed(spec)
        except (ParadigmInvalid, ValueError, OSError):
            continue
        report = ws.report_path("eig", p.sha256)
        if p.sha256 in exclude or not report.exists():
            continue
        eig = json.loads(report.read_text())
        if eig.get("paradigm_sha256") != p.sha256 or not eig.get("meets_floor"):
            continue
        if best is None or eig["bits_per_dollar"] > best["bits_per_dollar"]:
            best = {"spec": str(spec.resolve()), "sha256": p.sha256, "bits_per_dollar": eig["bits_per_dollar"]}
    return best


def seed_for(sha: str, purpose: str) -> int:
    """A seed fixed by the spec and the use: the agent cannot pick a lucky one."""
    return int(hashlib.sha256(f"{purpose}:{sha}".encode()).hexdigest()[:8], 16)


def _log(ws: Workspace, command: str, spec: Path, sha: Optional[str], outcome: str) -> None:
    ws.work.mkdir(parents=True, exist_ok=True)
    with ws.log.open("a") as f:
        f.write(json.dumps({"command": command, "spec": str(spec), "sha256": sha, "outcome": outcome,
                            "time": time.time()}) + "\n")


def _write_report(ws: Workspace, command: str, sha: str, report: Dict[str, Any]) -> Path:
    ws.reports.mkdir(parents=True, exist_ok=True)
    path = ws.report_path(command, sha)
    path.write_text(json.dumps(report, indent=2))
    return path


# --------------------------------------------------------------------------- commands

def preview_units(paradigm, seed: int = 0) -> list:
    """``distinct_trials`` random units of the paradigm's scene space, for a time estimate before
    there is a design."""
    from src.rsa_open.design import describe_unit, trial_units
    from src.rsa_open.paradigm import paradigm_likelihood, scene_pool

    rng = np.random.default_rng(seed)
    scenes = scene_pool(paradigm)
    scenes = [scenes[i] for i in rng.permutation(len(scenes))[:2000]]
    if paradigm.response.type == "bet":
        units = [{"objects": [list(o) for o in s.objects], "heard": s.utterance} for s in scenes]
    else:
        units = [describe_unit(scenes, u) for u in trial_units(scenes, paradigm_likelihood(paradigm))]
    need = paradigm.design.distinct_trials
    if len(units) < need:
        raise ToolRefused(f"the scene space gives {len(units)} trial units; distinct_trials is {need}")
    return [units[i] for i in rng.choice(len(units), need, replace=False)]


def time_report(ws: Workspace, paradigm, units: list) -> Dict[str, Any]:
    """The pilot's time estimate for these units, calibrated on the workspace's timing records."""
    from src.rsa_open.pilot import TIME_TOLERANCE
    from src.rsa_open.screens import design_units, trial_lists
    from src.rsa_open.timing import estimate

    lists = trial_lists(paradigm, design_units({"design": units}), seed=0, n_lists=20)
    t = estimate(lists, ws.timing_records())
    ratio = t["estimated_minutes"] / paradigm.design.minutes
    return {"estimated_minutes": t["estimated_minutes"], "declared_minutes": paradigm.design.minutes,
            "within_tolerance": abs(ratio - 1.0) <= TIME_TOLERANCE, "calibration_basis": t["calibration_basis"]}


def validate(ws: Workspace, spec: Path) -> Dict[str, Any]:
    from src.rsa_open.design import trial_units
    from src.rsa_open.paradigm import load_designed, paradigm_likelihood, scene_pool

    p = load_designed(spec)
    scenes = scene_pool(p)
    # The EIG tool's limit on trial units, checked here so a spec eig cannot score fails at once
    # (trial_units raises past it).
    n_units = len(trial_units(scenes, paradigm_likelihood(p)))
    return {"valid": True, "sha256": p.sha256, "scenes": len(scenes), "units": n_units, "cost_usd": p.cost_usd,
            "response": p.response.type, "link": p.response.link,
            "time": time_report(ws, p, preview_units(p, seed_for(p.sha256, "preview")))}


def eig_report(ws: Workspace, spec: Path, *, n_draws: Optional[int] = None,
               n_scenarios: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    from src.rsa_open.design import DEFAULT_DRAWS, DEFAULT_SCENARIOS, load_context, score_paradigm
    from src.rsa_open.paradigm import load_paradigm

    p = load_paradigm(spec)
    report = score_paradigm(p, load_context(ws.context), n_draws=n_draws or DEFAULT_DRAWS,
                            n_scenarios=n_scenarios or DEFAULT_SCENARIOS,
                            seed=seed_for(p.sha256, "eig") if seed is None else seed)
    report["time"] = time_report(ws, p, report["design"])
    return report


def default_theory(ctx) -> str:
    """The live theory a link check simulates from: the one with the most parameters (ties by name)."""
    return sorted(ctx.names, key=lambda n: (-len(ctx.theories[n].params), n))[0]


def link_check_report(ws: Workspace, spec: Path, *, theory: Optional[str] = None,
                      seed: Optional[int] = None, warmup: int = LINK_CHECK_WARMUP,
                      samples: int = LINK_CHECK_SAMPLES) -> Dict[str, Any]:
    from src.rsa.fit import FitSettings
    from src.rsa_open.design import load_context
    from src.rsa_open.fit import recovery_check
    from src.rsa_open.paradigm import load_paradigm, paradigm_likelihood, scene_pool

    p = load_paradigm(spec)
    ctx = load_context(ws.context)
    name = theory or default_theory(ctx)
    if name not in ctx.theories:
        raise ToolRefused(f"no live theory {name!r}; the live theories are {ctx.names}")
    link = paradigm_likelihood(p)
    truth = {**{k: float(np.mean(v)) for k, v in ctx.draws[name].items()}, **link.reference_params}
    seed = seed_for(p.sha256, "link_check") if seed is None else seed
    n_trials = p.design.n_participants * p.design.trials_per_participant
    settings = FitSettings(num_warmup=warmup, num_samples=samples, num_chains=2, seed=seed, dense_mass=True)
    report = recovery_check(ctx.root / "theories" / f"{name}.py", link, truth, n_trials=n_trials, seed=seed,
                            contexts=scene_pool(p), settings=settings)
    return {"paradigm_sha256": p.sha256, "theory": name, "truth": truth, **report}


def submit(ws: Workspace, spec: Path) -> Dict[str, Any]:
    from src.rsa_open.paradigm import load_designed

    p = load_designed(spec)
    problems = []
    eig_path, link_path = ws.report_path("eig", p.sha256), ws.report_path("link-check", p.sha256)
    if not eig_path.exists():
        problems.append("no eig report for this exact spec: run `eig` on it")
    elif not json.loads(eig_path.read_text()).get("meets_floor"):
        problems.append("its eig report is below the floor")
    if not link_path.exists():
        problems.append("no link-check report for this exact spec: run `link-check` on it")
    else:
        lc = json.loads(link_path.read_text())
        if not (lc.get("converged") and lc.get("recovered")):
            problems.append(f"its link check failed (converged={lc.get('converged')}, misses={lc.get('misses')})")
    if problems:
        raise ToolRefused("not submitted:\n" + "\n".join(f"- {x}" for x in problems))
    record = {"spec": str(Path(spec).resolve()), "sha256": p.sha256, "time": time.time()}
    ws.submission.write_text(json.dumps(record, indent=2))
    return record


# --------------------------------------------------------------------------- CLI

@dataclass
class _Common:
    workspace: Path
    """The designer workspace (the brief gives its path)."""
    spec: Path
    """The paradigm spec (a JSON file in the workspace's work/)."""


@dataclass
class Validate(_Common):
    """List every problem with a spec, or confirm it is valid."""


@dataclass
class Eig(_Common):
    """Score the paradigm's design under the live theories: bits, bits per dollar, the floor."""

    quick: bool = False
    """An approximate score (fewer draws and scenarios; minutes less): for exploring. Its report
    is saved apart, and submit needs a full eig of the spec."""


@dataclass
class LinkCheck(_Common):
    """The link's recovery test in this paradigm (several minutes: three NUTS fits)."""

    theory: Optional[str] = None
    """The live theory to simulate from (default: the first)."""


@dataclass
class Submit(_Common):
    """Submit the spec (needs a passing eig and link-check report of this exact spec)."""


Command = Union[
    Annotated[Validate, tyro.conf.subcommand("validate")],
    Annotated[Eig, tyro.conf.subcommand("eig")],
    Annotated[LinkCheck, tyro.conf.subcommand("link-check")],
    Annotated[Submit, tyro.conf.subcommand("submit")],
]
NAMES = {Validate: "validate", Eig: "eig", LinkCheck: "link-check", Submit: "submit"}


def report_name(command: str, report: Dict[str, Any]) -> str:
    """A quick eig's report is saved apart: submit reads only the full one."""
    return "eig-quick" if command == "eig" and report.get("quick") else command


def _summary(command: str, report: Dict[str, Any]) -> str:
    t = report.get("time") if command in ("validate", "eig") else None
    timing = "" if t is None else (
        f"; time: estimated {t['estimated_minutes']:.1f} min against your {t['declared_minutes']:g} "
        f"({'within' if t['within_tolerance'] else 'NOT within'} the pilot's 30%; calibrated on "
        f"{t['calibration_basis']})")
    if command == "validate":
        units = "" if report["units"] is None else f" ({report['units']} trial units)"
        return (f"valid: {report['scenes']} scenes{units}, {report['response']} via {report['link']}, "
                f"${report['cost_usd']:.2f}; sha256 {report['sha256'][:12]}{timing}")
    if command == "eig":
        line = (f"{report['joint_eig_bits']:.3f} of {report['max_bits']:.3f} bits "
                f"(p_correct {report['p_correct']:.2f}), ${report['cost_usd']:.2f}, "
                f"{report['bits_per_dollar']:.5f} bits/$")
        if report.get("quick"):
            line = "QUICK (approximate; submit needs a full eig): " + line
        if "floor_bits_per_dollar" in report:
            line += (f"; floor {report['floor_bits_per_dollar']:.5f}: "
                     f"{'MEETS' if report['meets_floor'] else 'BELOW'}")
        return line + timing
    if command == "link-check":
        return (f"link {report['link']} in this paradigm ({report['n_trials']} responses, theory "
                f"{report['theory']}): converged={report['converged']}, recovered={report['recovered']}, "
                f"misses={report['misses']}")
    return f"submitted {report['sha256'][:12]}"


def run(cmd: Any) -> Dict[str, Any]:
    """Run one command, under the budget; return its report (raises `ToolRefused` or `ParadigmInvalid`)."""
    from src.rsa_open.paradigm import ParadigmInvalid, load_paradigm

    command = NAMES[type(cmd)]
    ws = Workspace(Path(cmd.workspace))
    ws.check()
    used = len(ws.calls())
    if used >= MAX_TOOL_CALLS:
        raise ToolRefused(f"the tool budget ({MAX_TOOL_CALLS} calls) is spent: submit what you have or stop")
    try:
        sha = load_paradigm(cmd.spec).sha256
    except (ParadigmInvalid, OSError):
        sha = None
    try:
        if command == "validate":
            report = validate(ws, cmd.spec)
        elif command == "eig" and cmd.quick:
            report = eig_report(ws, cmd.spec, n_draws=QUICK_DRAWS, n_scenarios=QUICK_SCENARIOS)
            report["quick"] = True
        elif command == "eig":
            report = eig_report(ws, cmd.spec)
        elif command == "link-check":
            report = link_check_report(ws, cmd.spec, theory=cmd.theory)
        else:
            report = submit(ws, cmd.spec)
    except Exception as err:
        _log(ws, command, cmd.spec, sha, f"refused: {type(err).__name__}")
        raise
    _log(ws, command, cmd.spec, sha, "ok")
    if command in ("eig", "link-check"):
        _write_report(ws, report_name(command, report), report["paradigm_sha256"], report)
    report["calls_left"] = MAX_TOOL_CALLS - used - 1
    return report


def main(argv: Optional[list] = None) -> int:
    from src.rsa_open.paradigm import ParadigmInvalid

    cmd = tyro.cli(Command, args=argv)
    command = NAMES[type(cmd)]
    ws = Workspace(Path(cmd.workspace))
    try:
        report = run(cmd)
    except (ParadigmInvalid, ToolRefused, NotImplementedError, ValueError) as err:
        print(f"{command}: {err}")
        _print_time(ws)
        return 1
    print(f"{command}: {_summary(command, report)} ({report['calls_left']} tool calls left)")
    if command in ("eig", "link-check"):
        print(f"report: {ws.report_path(report_name(command, report), report['paradigm_sha256'])}")
    _print_time(ws)
    return 0


def _print_time(ws: Workspace) -> None:
    note = time_note(ws) if ws.inputs.is_dir() else None
    if note:
        print(note)


if __name__ == "__main__":
    sys.exit(main())
