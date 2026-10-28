"""Estimated cost of agent calls the backend did not price (run C: 5 of 32 agent records had
``cost_usd`` 0, among them a designer session of 4.1 M tokens, because opencode reported no price).

Each unpriced record is priced at its model's own rate in the same log: the priced records' cost
per token, input and output separately when there are enough of them (least squares, never
negative), else per token overall. A model with no priced record stays unpriced. The estimates go to
``token_usage_estimates.json`` beside the summary, marked as estimates; the records are not changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def _inputs(r) -> float:
    return float(r.input_tokens + r.cache_read_tokens + r.cache_write_tokens)


def _outputs(r) -> float:
    return float(r.output_tokens + r.reasoning_tokens)


def estimate_unpriced(records: Sequence) -> Dict:
    priced = [r for r in records if r.cost_usd and r.total_tokens]
    unpriced = [r for r in records if not r.cost_usd and r.total_tokens and not r.usage_missing]
    out: Dict = {"estimated": [], "unpriced_models": [], "estimated_usd": 0.0}
    for model in sorted({r.model or "?" for r in unpriced}):
        mine = [r for r in priced if (r.model or "?") == model]
        todo = [r for r in unpriced if (r.model or "?") == model]
        if not mine:
            out["unpriced_models"].append({"model": model, "records": len(todo),
                                           "tokens": int(sum(r.total_tokens for r in todo))})
            continue
        rate_in: Optional[float] = None
        if len(mine) >= 3:
            a = np.array([[_inputs(r), _outputs(r)] for r in mine])
            coef, *_ = np.linalg.lstsq(a, np.array([r.cost_usd for r in mine]), rcond=None)
            if np.all(coef >= 0):
                rate_in, rate_out = float(coef[0]), float(coef[1])
        if rate_in is None:
            per = sum(r.cost_usd for r in mine) / sum(r.total_tokens for r in mine)
            rate_in = rate_out = per
        for r in todo:
            usd = rate_in * _inputs(r) + rate_out * _outputs(r)
            out["estimated"].append({"source": r.source, "model": model, "tokens": int(r.total_tokens),
                                     "estimated_usd": round(usd, 4), "timestamp": r.timestamp})
            out["estimated_usd"] += usd
    out["estimated_usd"] = round(out["estimated_usd"], 2)
    return out


def write_estimates(run_dir: Path, records: Sequence) -> Dict:
    est = estimate_unpriced(records)
    (Path(run_dir) / "token_usage_estimates.json").write_text(json.dumps(est, indent=1))
    if est["estimated"] or est["unpriced_models"]:
        print(f"  [tokens]   unpriced by the backend: {len(est['estimated'])} record(s) estimated at "
              f"${est['estimated_usd']:.2f} from their models' priced records"
              + "".join(f"; {m['model']}: {m['records']} record(s), {m['tokens']:,} tokens, no priced record to "
                        "estimate from" for m in est["unpriced_models"]), flush=True)
    return est
