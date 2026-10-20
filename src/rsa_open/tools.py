"""The designer's tools (docs/auto_open/PLAN.md §4): a CLI the designer agent runs in its sandbox.

    <python> -m src.rsa_open.tools validate   --workspace <ws> --spec <ws>/work/p.json
    <python> -m src.rsa_open.tools eig        --workspace <ws> --spec <ws>/work/p.json
    <python> -m src.rsa_open.tools link-check --workspace <ws> --spec <ws>/work/p.json
    <python> -m src.rsa_open.tools submit     --workspace <ws> --spec <ws>/work/p.json

**The workspace** (`Workspace`) is one designer session's directory, laid out by the harness
(`src.rsa_open.designer`):
- ``context/``: the design context (`src.rsa_open.design`), with the closed paradigm's reference
  score already computed; read-only to the agent;
- ``inputs/``: the brief, the spec's JSON schema, the link library and the chain's earlier
  paradigms; read-only;
- ``work/``: the agent's own directory (its specs, ``reports/``, ``tool_log.jsonl`` and
  ``submission.json``).

**The commands:**
- ``validate``: the spec's problems, all at once (`src.rsa_open.paradigm`);
- ``eig``: the paradigm's design under the live theories, its bits per dollar and the floor;
- ``link-check``: the recovery test (`src.rsa_open.fit.recovery_check`) of the spec's link **in
  this paradigm**: its scene space, its number of responses, its scale, simulated from a live
  theory at its posterior mean and the link at its reference values;
- ``submit``: records the spec as the session's submission, once it has an ``eig`` report that
  meets the floor and a ``link-check`` report that recovers, both of this exact spec (sha256).

Every report is written to ``work/reports/<command>_<sha12>.json``. ``submit`` only records:
the harness then re-scores the spec with a seed of its own, reruns the link check, and pilots
it (`src.rsa_open.designer`); the agent's reports are never trusted as the gate.

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
# Link check: NUTS settings and the replicates of the library's recovery test.
LINK_CHECK_WARMUP = 500
LINK_CHECK_SAMPLES = 500


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

def validate(ws: Workspace, spec: Path) -> Dict[str, Any]:
    from src.rsa_open.paradigm import load_paradigm, scene_pool

    p = load_paradigm(spec)
    return {"valid": True, "sha256": p.sha256, "scenes": len(scene_pool(p)), "cost_usd": p.cost_usd,
            "response": p.response.type, "link": p.response.link}


def eig_report(ws: Workspace, spec: Path, *, n_draws: Optional[int] = None,
               n_scenarios: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    from src.rsa_open.design import DEFAULT_DRAWS, DEFAULT_SCENARIOS, load_context, score_paradigm
    from src.rsa_open.paradigm import load_paradigm

    p = load_paradigm(spec)
    return score_paradigm(p, load_context(ws.context), n_draws=n_draws or DEFAULT_DRAWS,
                          n_scenarios=n_scenarios or DEFAULT_SCENARIOS,
                          seed=seed_for(p.sha256, "eig") if seed is None else seed)


def link_check_report(ws: Workspace, spec: Path, *, theory: Optional[str] = None,
                      seed: Optional[int] = None, warmup: int = LINK_CHECK_WARMUP,
                      samples: int = LINK_CHECK_SAMPLES) -> Dict[str, Any]:
    from src.rsa.fit import FitSettings
    from src.rsa_open.design import load_context
    from src.rsa_open.fit import recovery_check
    from src.rsa_open.paradigm import load_paradigm, paradigm_likelihood, scene_pool

    p = load_paradigm(spec)
    ctx = load_context(ws.context)
    name = theory or ctx.names[0]
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
    from src.rsa_open.paradigm import load_paradigm

    p = load_paradigm(spec)
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


def _summary(command: str, report: Dict[str, Any]) -> str:
    if command == "validate":
        return (f"valid: {report['scenes']} scenes, {report['response']} via {report['link']}, "
                f"${report['cost_usd']:.2f}; sha256 {report['sha256'][:12]}")
    if command == "eig":
        line = (f"{report['joint_eig_bits']:.3f} of {report['max_bits']:.3f} bits "
                f"(p_correct {report['p_correct']:.2f}), ${report['cost_usd']:.2f}, "
                f"{report['bits_per_dollar']:.5f} bits/$")
        if "floor_bits_per_dollar" in report:
            line += (f"; floor {report['floor_bits_per_dollar']:.5f}: "
                     f"{'MEETS' if report['meets_floor'] else 'BELOW'}")
        return line
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
        _write_report(ws, command, report["paradigm_sha256"], report)
    report["calls_left"] = MAX_TOOL_CALLS - used - 1
    return report


def main(argv: Optional[list] = None) -> int:
    from src.rsa_open.paradigm import ParadigmInvalid

    cmd = tyro.cli(Command, args=argv)
    command = NAMES[type(cmd)]
    try:
        report = run(cmd)
    except (ParadigmInvalid, ToolRefused, NotImplementedError, ValueError) as err:
        print(f"{command}: {err}")
        return 1
    print(f"{command}: {_summary(command, report)} ({report['calls_left']} tool calls left)")
    if command in ("eig", "link-check"):
        print(f"report: {Workspace(Path(cmd.workspace)).report_path(command, report['paradigm_sha256'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
