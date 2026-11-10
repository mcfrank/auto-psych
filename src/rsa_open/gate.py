"""The feasibility gate (`docs/auto_open/PREREG_V2.md` §4): the registered criteria, computed from
each stage-1 chain's run tree alone. Stage 1 is open chain 1 and fixed chain 1; the gate passes only
if both do.

    python -m src.rsa_open.gate --run-dir <cells>/open_chain_0 <cells>/fixed_chain_0 [--json]

This file is registered (its sha256 is in the pre-registration), so the gate is fixed before any
data it judges. It reads only what the chain writes: each experiment's frozen design record, its
collection record (``data/participants.json``, aggregates), its loop history and export, the
chain's stage times (``stage_times.jsonl``) and agent spend (``token_usage.jsonl``). It never
compares the two arms.

The items (the definitions accepted by the PI, 2026-10-29; item 5 as revised, 2026-11-09/10,
`PREREG_V2.md`), per experiment 1-3 (``EXPERIMENTS``):
1. **Design:** the experiment froze a paradigm (``design/frozen/record.json``): for the open arm within
   the designer's two cycles, since both failing stops the chain without one; for the fixed arm its
   frozen spec passed the harness's checks.
2. **Data:** among recruited participants, final comprehension and catch trials passed each at
   least ``FLOORS``; exclusions at most ``MAX_EXCLUSIONS``; people giving the same answer on
   nearly every test trial (the monitor's flag: at least 90% of their test answers the same) at
   most ``MAX_SAME_ANSWER`` of those readable; and the data parsed (the experiment's block was
   written and the loop ran on it).
3. **Information:** the data carry signal the theories capture: the loop's incumbent at its
   first scoring (the best of the theories carried in, before any new one) beats the chance
   theory (`src.rsa_open.loop.orchestrator.CHANCE_THEORY`: no sensitivity to the display, its
   link fitted) on the new paradigm by more than ``SEPARATION_SE`` unit-clustered SEs, both
   scored by the loop's grouped cross-validation (``loop/newest_block_cv.json``). How far the
   carried theories are apart is reported, not judged: the loop prunes them to near-ties (run B's
   experiments 2-3: 0.4 and 1.0 SEs).
4. **Loop:** at least two trusted theories at the end and an export (no harness failure).
5. **Time:** the experiment ran within ``MAX_EXPERIMENT_HOURS`` (PI 2026-11-09): from its design's
   first start to its loop's last finish, less its collection stage (from its first start to its
   finish: recruitment, which the 24-hour window bounds, the deploy, and any wait for the PI's
   review); queue waits and failures count.
And for the three experiments together:
4b. **Admissions:** at least one new theory admitted over experiments 1-3 (run B's experiment 3
   admitted none: its loop stopped after two stale rounds).
Reported, not judged (PI 2026-11-10): agent spend (every priced call plus the estimate for unpriced
ones, `src.rsa_open.spend`), the chain's total running time, and participant payments, which the
budget bounds by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import tyro

EXPERIMENTS = 3
FLOORS = {"comprehension_final": 0.8, "catch_pass": 0.8}
MAX_EXCLUSIONS = 0.25
MAX_SAME_ANSWER = 0.10
SEPARATION_SE = 4.0
MIN_TRUSTED_AT_END = 2
MAX_EXPERIMENT_HOURS = 48.0  # PREREG_V2.md §4, item 5 (PI 2026-11-09)


def _read(path: Path) -> Optional[Any]:
    return json.loads(path.read_text()) if path.exists() else None


def _item(passed: bool, detail: str, **values) -> Dict[str, Any]:
    return {"passed": bool(passed), "detail": detail, **values}


def design_item(exp: Path) -> Dict[str, Any]:
    record = _read(exp / "design" / "frozen" / "record.json")
    if record is None:
        return _item(False, "no frozen paradigm")
    return _item(True, f"froze paradigm {record['paradigm_sha256'][:12]} (submitted by {record.get('submitted_by')})",
                 joint_eig_bits=record.get("joint_eig_bits"))


def data_item(exp: Path) -> Dict[str, Any]:
    rec = _read(exp / "data" / "participants.json")
    if rec is None or rec.get("collection") != "live":
        return _item(False, "no live collection record")
    if rec.get("smoke_test"):
        return _item(False, "a smoke test, never judged")
    problems = [f"{k} {rec.get(k)} < {floor}" for k, floor in FLOORS.items()
                if rec.get(k) is None or rec[k] < floor]
    if rec.get("exclusion_rate") is None or rec["exclusion_rate"] > MAX_EXCLUSIONS:
        problems.append(f"exclusions {rec.get('exclusion_rate')} > {MAX_EXCLUSIONS}")
    same = (rec.get("same_answer") or {}).get("share")
    if same is None or same > MAX_SAME_ANSWER:
        problems.append(f"same answer on nearly every trial {same} > {MAX_SAME_ANSWER}")
    if not (exp / "data" / "block.json").exists() or not (exp / "loop" / "history.json").exists():
        problems.append("the data were not parsed into a block the loop ran on")
    values = {k: rec.get(k) for k in (*FLOORS, "exclusion_rate", "recruited", "kept")}
    return _item(not problems, "; ".join(problems) or "all within bounds", same_answer_share=same, **values)


def carried_spread(first: Mapping[str, Any], block: str) -> Optional[Dict[str, Any]]:
    """Reported, not judged: how far behind the best the furthest trusted carried theory is."""
    rows = {n: d["by_block"][block] for n, d in first["standing"].items()
            if d.get("trusted") and block in d.get("by_block", {})}
    if len(rows) < 2:
        return None
    worst = max(rows, key=lambda n: rows[n]["behind"])
    behind, dse = rows[worst]["behind"], rows[worst]["dse"]
    return {"theories": len(rows), "furthest": worst, "behind": behind, "dse": dse,
            "ratio": behind / dse if dse > 0 else float("inf")}


def information_item(exp: Path) -> Dict[str, Any]:
    from src.models.clustered_se import cluster_dse

    history, cv = _read(exp / "loop" / "history.json"), _read(exp / "loop" / "newest_block_cv.json")
    if not history or cv is None:
        return _item(False, "no loop history or newest_block_cv.json")
    first, block = history[0], cv["block"]
    spread = carried_spread(first, block)
    best = first["best_model"]
    if cv.get("chance") is None:
        return _item(False, f"the chance theory was not scored ({cv.get('chance_problem')})", carried=spread)
    if not first["standing"][best].get("trusted") or best not in cv["theories"]:
        return _item(False, f"the first incumbent {best} has no trusted out-of-fold score", carried=spread)
    a, b = np.asarray(cv["theories"][best]), np.asarray(cv["chance"])
    diff, dse = float(a.sum() - b.sum()), float(cluster_dse(a, b, np.asarray(cv["units"])))
    ratio = diff / dse if dse > 0 else (float("inf") if diff > 0 else 0.0)
    return _item(ratio > SEPARATION_SE, f"{best} is {diff:.1f} nats ahead of chance on {block} ({ratio:.1f} clustered "
                                        f"SEs; more than {SEPARATION_SE:g} needed)"
                                        + ("" if cv.get("chance_converged") else "; the chance theory's CV did not converge"),
                 incumbent=best, ahead_of_chance=diff, dse=dse, ratio=ratio,
                 chance_converged=cv.get("chance_converged"), carried=spread)


def loop_item(exp: Path) -> Dict[str, Any]:
    history, export = _read(exp / "loop" / "history.json"), _read(exp / "loop" / "export.json")
    if not history or export is None:
        return _item(False, "the loop did not finish (no export)")
    admitted = sorted({e["name"] for h in history for e in h["events"] if e.get("outcome") == "admitted"})
    last = history[-1]
    trusted = sorted(n for n, d in last["standing"].items() if d.get("trusted") and n in export["live"])
    problems = []
    if len(trusted) < MIN_TRUSTED_AT_END:
        problems.append(f"{len(trusted)} trusted theories at the end (at least {MIN_TRUSTED_AT_END})")
    return _item(not problems, "; ".join(problems) or f"{len(admitted)} admitted, {len(trusted)} trusted at the end",
                 admitted=len(admitted), trusted_at_end=len(trusted), stopped_early=export.get("stopped_early"))


def _marks(run_dir: Path) -> Optional[List[Dict[str, Any]]]:
    path = run_dir / "stage_times.jsonl"
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _at(marks, n, stage, event, pick):
    times = [m["at"] for m in marks if m["experiment"] == n and m["stage"] == stage and m["event"] == event]
    return pick(times) if times else None


def running_hours(run_dir: Path, first: int = 1, last: int = EXPERIMENTS) -> Optional[float]:
    """Hours from experiment ``first``'s design start to experiment ``last``'s loop finish, less each
    experiment's collection stage; None when the stage times do not cover them."""
    marks = _marks(run_dir)
    if marks is None:
        return None
    start, end = _at(marks, first, "design", "started", min), _at(marks, last, "loop", "finished", max)
    if start is None or end is None:
        return None
    collecting = 0.0
    for n in range(first, last + 1):
        a, b = _at(marks, n, "collect", "started", min), _at(marks, n, "collect", "finished", max)
        if a is None or b is None:
            return None
        collecting += b - a
    return (end - start - collecting) / 3600


def time_item(run_dir: Path, n: int) -> Dict[str, Any]:
    hours = running_hours(run_dir, first=n, last=n)
    if hours is None:
        return _item(False, "the stage times do not cover this experiment's design, collection and loop")
    return _item(hours <= MAX_EXPERIMENT_HOURS, f"{hours:.1f} h running (at most {MAX_EXPERIMENT_HOURS:g})",
                 running_hours=hours)


def agent_usd(run_dir: Path) -> Optional[float]:
    from src.rsa_open.spend import estimate_unpriced
    from src.runtime.token_usage import read_usage_log

    path = run_dir / "token_usage.jsonl"
    if not path.exists():
        return None
    records = read_usage_log(path)
    return sum(r.cost_usd for r in records if r.cost_usd) + estimate_unpriced(records)["estimated_usd"]


def participant_usd(exp: Path) -> Optional[float]:
    from src.rsa_open.paradigm import load_paradigm

    path = exp / "design" / "frozen" / "paradigm.json"
    if not path.exists():
        return None
    return load_paradigm(path).cost_usd


def reported(run_dir: Path) -> Dict[str, Any]:
    """What the gate reports and does not judge: the chain's running time, agent spend, participant payments."""
    hours, usd = running_hours(run_dir), agent_usd(run_dir)
    paid = [participant_usd(run_dir / f"experiment{n}") for n in range(1, EXPERIMENTS + 1)]
    return {"running_hours": hours, "agent_usd": usd,
            "participant_usd": sum(p for p in paid if p is not None) if any(p is not None for p in paid) else None}


def admissions_item(run_dir: Path) -> Dict[str, Any]:
    admitted = sum(item["admitted"] for n in range(1, EXPERIMENTS + 1)
                   if (item := loop_item(run_dir / f"experiment{n}")).get("admitted") is not None)
    return _item(admitted >= 1, f"{admitted} new theories admitted over experiments 1-{EXPERIMENTS}", admitted=admitted)


def evaluate(run_dir: Path) -> Dict[str, Any]:
    """One chain's items (§4): each experiment's 1-5, and admissions over the three."""
    run_dir = Path(run_dir)
    out: Dict[str, Any] = {"run_dir": str(run_dir), "experiments": {}}
    for n in range(1, EXPERIMENTS + 1):
        exp = run_dir / f"experiment{n}"
        out["experiments"][n] = {"1_design": design_item(exp), "2_data": data_item(exp),
                                 "3_information": information_item(exp), "4_loop": loop_item(exp),
                                 "5_time": time_item(run_dir, n)}
    out["4b_admissions"] = admissions_item(run_dir)
    out["reported"] = reported(run_dir)
    out["passed"] = (all(i["passed"] for e in out["experiments"].values() for i in e.values())
                     and out["4b_admissions"]["passed"])
    return out


def evaluate_stage(run_dirs: List[Path]) -> Dict[str, Any]:
    """The gate: every stage-1 chain's items; it passes only if every chain passes."""
    if not run_dirs:
        raise ValueError("no chain to judge")
    chains = [evaluate(d) for d in run_dirs]
    return {"chains": chains, "passed": all(c["passed"] for c in chains)}


def lines(report: Dict[str, Any]) -> List[str]:
    mark = {True: "PASS", False: "FAIL"}
    out = [f"Feasibility gate (PREREG_V2.md §4): {mark[report['passed']]}"]
    for chain in report["chains"]:
        out += ["", f"{chain['run_dir']}: {mark[chain['passed']]}"]
        for n, items in chain["experiments"].items():
            for key, item in items.items():
                out.append(f"experiment {n}, {key}: {mark[item['passed']]}: {item['detail']}")
        a = chain["4b_admissions"]
        out.append(f"4b_admissions: {mark[a['passed']]}: {a['detail']}")
        r = chain["reported"]
        out.append("reported, not judged: " + "; ".join([
            "running time not covered" if r["running_hours"] is None else f"{r['running_hours']:.0f} h running",
            "no agent spend record" if r["agent_usd"] is None else f"${r['agent_usd']:.2f} of agents",
            "" if r["participant_usd"] is None else f"participant payments ${r['participant_usd']:.2f}"]).rstrip("; "))
    return out


@dataclass
class Args:
    run_dir: List[Path]
    """Each stage-1 chain's run directory (open chain 1 and fixed chain 1)."""
    json: bool = False


def main(args: Args) -> Dict[str, Any]:
    report = evaluate_stage(list(args.run_dir))
    print(json.dumps(report, indent=1, default=float) if args.json else "\n".join(lines(report)))
    return report


if __name__ == "__main__":
    main(tyro.cli(Args))
