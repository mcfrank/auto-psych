"""Every chain of a campaign at a glance (`DISTRIBUTED_DESIGN.md` v2, §0): what each is doing, what
stopped it, which studies are recruiting, and what has been spent.

    python -m src.rsa_open.status --work-root $SCRATCH/auto-psych/rsa_open_live [--squeue] [--json]

It reads only the run trees and cell records (no Prolific or Firebase call), so it is quick and
safe to run at any time. ``--squeue`` adds the user's Slurm jobs, from one ``squeue`` call. A chain
with no STOP whose files have not changed for ``--quiet-hours`` is flagged QUIET: something may
be stuck, as open chain 1's crash sat unnoticed for 5 h (2026-11-01).

Layout read (as `scripts/rsa/slurm/open_live.sbatch` writes it): ``<work root>/cells/<arm>_chain_<k>
[_smoke]/`` holds STOP, DONE and ``agent_tree_id``; the chain's run directory is
``<agent trees>/<agent_tree_id>/repo/_runs/chain_<k>`` (agent trees default to the work root's
sibling ``agent_trees``, as `_env.sh` sets them).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import tyro

QUIET_HOURS = 2.0
CELL_RE = re.compile(r"^(open|fixed)_chain_(\d+)(_smoke)?$")


def _read(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _newest_mtime(root: Path, depth: int = 3) -> Optional[float]:
    """The newest modification time under ``root``, to ``depth`` levels (fit caches included)."""
    newest = None
    stack = [(root, 0)]
    while stack:
        here, level = stack.pop()
        try:
            entries = list(os.scandir(here))
        except OSError:
            continue
        for e in entries:
            try:
                m = e.stat(follow_symlinks=False).st_mtime
            except OSError:
                continue
            newest = m if newest is None else max(newest, m)
            if level < depth and e.is_dir(follow_symlinks=False):
                stack.append((Path(e.path), level + 1))
    return newest


def _stage_now(run_dir: Path) -> Optional[Dict[str, Any]]:
    """The latest stage event in ``stage_times.jsonl``: (experiment, stage, started/finished, when)."""
    path = run_dir / "stage_times.jsonl"
    if not path.exists():
        return None
    marks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return marks[-1] if marks else None


def experiment_status(exp: Path) -> Dict[str, Any]:
    n = int(exp.name.removeprefix("experiment"))
    out: Dict[str, Any] = {"experiment": n}
    record = _read(exp / "design" / "frozen" / "record.json")
    paradigm = _read(exp / "design" / "frozen" / "paradigm.json")
    if record is not None:
        out["design"] = {"submitted_by": record.get("submitted_by"),
                         "response": (paradigm or {}).get("response", {}).get("type"),
                         "n_participants": (paradigm or {}).get("design", {}).get("n_participants"),
                         "pay_usd": (paradigm or {}).get("design", {}).get("pay_usd")}
    elif (exp / "design").exists():
        out["design"] = {"in_progress": True, "cycles": len(list((exp / "design").glob("cycle_*")))}
    manifest = _read(exp / "deployment" / "deployment_manifest.json")
    if manifest is not None:
        started = _read(exp / "deployment" / "collection_started.json")
        ended = _read(exp / "deployment" / "collection_ended.json")
        snap = _read(exp / "deployment" / "monitor.json") or {}
        out["study"] = {"mode": manifest.get("prolific_mode"), "study_id": manifest.get("prolific_study_id"),
                        "recruiting": bool(started) and ended is None and manifest.get("prolific_mode") == "live",
                        "hours_open": (time.time() - started["at"]) / 3600 if started and ended is None else None,
                        "completed": (ended or {}).get("completed", snap.get("recruited")),
                        "target": (ended or {}).get("target")}
    people = _read(exp / "data" / "participants.json")
    if people is not None:
        out["data"] = {k: people.get(k) for k in ("recruited", "kept", "exclusion_rate")}
        out["data"]["enjoyment"] = (people.get("enjoyment") or {}).get("mean")
    history = _read(exp / "loop" / "history.json")
    if history:
        out["loop"] = {"rounds_scored": max(h.get("round", -1) for h in history) + 1, "best": history[-1].get("best_model"),
                       "done": (exp / "loop" / "export.json").exists(),
                       "live": len(history[-1].get("live", []))}
    elif (exp / "loop").exists():
        out["loop"] = {"rounds_scored": 0, "done": False, "starting": True}
    return out


def chain_status(cell_dir: Path, agent_trees: Path, quiet_hours: float = QUIET_HOURS,
                 now: Optional[float] = None) -> Dict[str, Any]:
    now = time.time() if now is None else now
    m = CELL_RE.match(cell_dir.name)
    out: Dict[str, Any] = {"cell": cell_dir.name, "arm": m.group(1) if m else None,
                           "chain": int(m.group(2)) if m else None, "smoke": bool(m and m.group(3))}
    stop = cell_dir / "STOP"
    out["stop"] = stop.read_text().strip() if stop.exists() else None
    out["done"] = (cell_dir / "DONE").exists()
    tree_id = (cell_dir / "agent_tree_id").read_text().strip() if (cell_dir / "agent_tree_id").exists() else None
    run_dir = agent_trees / tree_id / "repo" / "_runs" / f"chain_{out['chain']}" if tree_id and m else None
    out["run_dir"] = str(run_dir) if run_dir else None
    if run_dir is None or not run_dir.exists():
        out["state"] = "stopped" if out["stop"] else "not started"
        out["experiments"] = []
        return out
    exps = sorted(run_dir.glob("experiment[0-9]*"), key=lambda p: int(p.name.removeprefix("experiment")))
    out["experiments"] = [experiment_status(e) for e in exps]
    out["stage"] = _stage_now(run_dir)
    newest = _newest_mtime(run_dir)
    out["idle_hours"] = None if newest is None else (now - newest) / 3600
    from src.rsa_open.gate import agent_usd

    try:
        out["agent_usd"] = agent_usd(run_dir)
    except Exception:  # noqa: BLE001 (a status line never fails on a record it can't read)
        out["agent_usd"] = None
    # live studies at their paradigm's N, pay and Prolific's fee; a smoke test recruits fewer, so it isn't counted
    out["participant_usd"] = None if out["smoke"] else round(sum(
        (e.get("design", {}).get("n_participants") or 0) * (e.get("design", {}).get("pay_usd") or 0) * 4 / 3
        for e in out["experiments"] if e.get("study", {}).get("mode") == "live"), 2)
    if out["stop"]:
        out["state"] = "done" if out["done"] else "stopped"
    elif out["idle_hours"] is not None and out["idle_hours"] > quiet_hours:
        out["state"] = "QUIET"
    else:
        out["state"] = "running"
    return out


def squeue_jobs() -> List[Dict[str, str]]:
    """The user's Slurm jobs of the campaign, from one ``squeue`` call (empty without Slurm)."""
    try:
        text = subprocess.run(["squeue", "--me", "-h", "-o", "%i|%j|%T|%M|%P|%R"], capture_output=True, text=True,
                              timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    keys = ("job", "name", "state", "elapsed", "partition", "reason")
    return [dict(zip(keys, line.split("|"))) for line in text.splitlines()
            if line.count("|") == 5 and line.split("|")[1].startswith("rsa_open")]


def campaign_status(work_root: Path, agent_trees: Optional[Path] = None, *, quiet_hours: float = QUIET_HOURS,
                    with_squeue: bool = False, now: Optional[float] = None) -> Dict[str, Any]:
    work_root = Path(work_root)
    agent_trees = Path(agent_trees) if agent_trees else work_root.parent / "agent_trees"
    cells = sorted(p for p in (work_root / "cells").glob("*_chain_*") if p.is_dir())
    chains = [chain_status(c, agent_trees, quiet_hours, now) for c in cells]
    recruiting = [{"cell": c["cell"], "experiment": e["experiment"], **e["study"]}
                  for c in chains for e in c["experiments"] if e.get("study", {}).get("recruiting")]
    out = {"work_root": str(work_root), "at": time.strftime("%Y-%m-%d %H:%M %Z"), "chains": chains,
           "recruiting": recruiting,
           "agent_usd": round(sum(c.get("agent_usd") or 0 for c in chains), 2),
           "participant_usd": round(sum(c.get("participant_usd") or 0 for c in chains), 2)}
    if with_squeue:
        out["jobs"] = squeue_jobs()
    return out


def _experiment_line(e: Dict[str, Any]) -> str:
    parts = [f"exp{e['experiment']}"]
    d = e.get("design") or {}
    if d.get("in_progress"):
        parts.append(f"designing (cycle {d['cycles']})")
    elif d:
        parts.append(f"{d.get('response')}, N {d.get('n_participants')}")
    s = e.get("study") or {}
    if s.get("recruiting"):
        parts.append(f"RECRUITING {s.get('completed') or 0}/{s.get('target') or '?'}, {s['hours_open']:.1f} h open")
    elif s.get("mode") == "test":
        parts.append("test deploy")
    data = e.get("data")
    if data:
        parts.append(f"kept {data.get('kept')} of {data.get('recruited')}")
    loop = e.get("loop")
    if loop:
        if loop.get("done"):
            parts.append(f"loop done, best {loop.get('best')}")
        elif loop.get("starting"):
            parts.append("loop: starting fits")
        else:
            parts.append(f"loop: {loop['rounds_scored']} round(s) scored, best {loop.get('best')}, {loop['live']} live")
    return ", ".join(parts)


def lines(report: Dict[str, Any]) -> List[str]:
    out = [f"Campaign {report['work_root']} at {report['at']}: agents ${report['agent_usd']:.2f}, "
           f"participants about ${report['participant_usd']:.2f}"]
    for c in report["chains"]:
        stage = c.get("stage")
        now = (f"exp{stage['experiment']} {stage['stage']} {stage['event']}" if stage else "no stage yet")
        idle = "" if c.get("idle_hours") is None else f", last change {c['idle_hours']:.1f} h ago"
        out.append(f"{c['cell']}: {c['state'].upper() if c['state'] in ('QUIET', 'stopped') else c['state']}; "
                   f"{now}{idle}; agents ${c.get('agent_usd') or 0:.2f}")
        if c.get("stop"):
            out.append(f"  STOP {c['stop']}")
        for e in c["experiments"]:
            out.append("  " + _experiment_line(e))
    if report["recruiting"]:
        out.append("Recruiting: " + "; ".join(f"{r['cell']} exp{r['experiment']} ({r['study_id']})"
                                              for r in report["recruiting"]))
    for j in report.get("jobs", []):
        out.append(f"job {j['job']} {j['name']} {j['state']} {j['elapsed']} on {j['partition']} ({j['reason']})")
    return out


@dataclass
class Args:
    work_root: Path
    agent_trees: Optional[Path] = None
    """Default: the work root's sibling ``agent_trees`` (as `scripts/rsa/slurm/_env.sh`)."""
    quiet_hours: float = QUIET_HOURS
    squeue: bool = False
    json: bool = False


def main(args: Args) -> Dict[str, Any]:
    report = campaign_status(args.work_root, args.agent_trees, quiet_hours=args.quiet_hours, with_squeue=args.squeue)
    print(json.dumps(report, indent=1, default=float) if args.json else "\n".join(lines(report)))
    return report


if __name__ == "__main__":
    main(tyro.cli(Args))
