"""One page over a finished RSA sweep (several loop cells): selection vs held-out
generalisation, recovery distances, cost.

    uv run python -m src.rsa.sweep_report --sweep data/rsa/sherlock_run1

Reads only the per-cell files a sweep brings back (handoff section 6):
``history.json``, ``export.json``, ``heldout/heldout.csv``,
``recovery/recovery.json``, ``attempted_hypotheses.jsonl``,
``token_usage_summary.json``, ``cell.json``, plus the sweep's
``run_notes.json`` (numbers measured on the cluster that no per-cell file
holds: wall time, peak memory, the best seed's distance to a recovery cell's
ground truth). Writes ``<sweep>/overview.html``, which links each cell's own
``report.html`` as ``cells/<cell>.html`` (the published layout).

"In-sample" is each model's ELPD-LOO at the last scored step before the
end-of-run prune, relative to the best; "held-out" is its held-out lpd minus
the best starting model's (src.rsa.evaluate_heldout), with the clustered SE.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import tyro

TEMPLATE = Path(__file__).with_name("sweep_report_template.html")


def _first_sentence(text: str, limit: int = 320) -> str:
    text = " ".join(text.split())
    end = text.find(". ")
    out = text if end < 0 else text[: end + 1]
    return out if len(out) <= limit else out[: limit - 1].rstrip() + "…"


def _heldout(cell_dir: Path) -> Dict[str, dict]:
    path = cell_dir / "heldout" / "heldout.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: was the cell scored?")
    out = {}
    for row in csv.DictReader(path.open()):
        out[row["model"]] = dict(
            lpd=float(row["lpd"]), diff=float(row["diff_vs_best_seed"]), se=float(row["se_diff_clustered"]),
            by_source={k[4:-1]: float(v) for k, v in row.items() if k.startswith("lpd[")},
        )
    return out


def _held_for(name: str, held: Dict[str, dict], live: set) -> dict | None:
    for label in ([name] if name in live else []) + [f"pruned:{name}", f"seed:{name}", name]:
        if label in held:
            return held[label]
    return None


def cell_bundle(cell_dir: Path, notes: Dict[str, Any]) -> dict:
    cell = json.loads((cell_dir / "cell.json").read_text())
    history = json.loads((cell_dir / "history.json").read_text())
    export = json.loads((cell_dir / "export.json").read_text())
    held = _heldout(cell_dir)
    live, best = set(export["live"]), export["best_model"]
    ledger = [json.loads(line) for line in (cell_dir / "attempted_hypotheses.jsonl").read_text().splitlines() if line.strip()]
    hypotheses = {}
    for e in ledger:
        if e["outcome"] == "admitted" or e["name"] not in hypotheses:
            hypotheses[e["name"]] = e["hypothesis"]
    seeds = {e["name"] for e in history[0]["events"] if e.get("outcome") == "seeded"}
    pre = history[-2]["standing"] if len(history) > 1 else history[-1]["standing"]
    top = max(s["elpd_loo"] for s in pre.values())
    models = []
    for name, s in pre.items():
        h = _held_for(name, held, live)
        if h is None:
            continue
        status = "exported" if name == best else "live" if name in live else "pruned"
        models.append(dict(
            name=name, status=status, seed=name in seeds,
            insample=s["elpd_loo"] - top, heldout=h["diff"], se=h["se"], by_source=h["by_source"],
            trusted=bool(s["loo_reliable"] and s["converged"]),
            hypothesis=_first_sentence(hypotheses.get(name, "")),
        ))
    counts = {k: sum(1 for e in ledger if e["outcome"] == k) for k in ("admitted", "rejected", "pruned")}
    usage = json.loads((cell_dir / "token_usage_summary.json").read_text())
    rec_path = cell_dir / "recovery" / "recovery.json"
    recovery = None
    if rec_path.exists():
        rec = json.loads(rec_path.read_text())
        recovery = dict(
            ground_truth=Path(rec["ground_truth"]).stem, exported=rec["best_model"],
            exported_rmse=rec["best_pool_rmse_to_gt"], live_rmse=rec["live_pool_rmse_to_gt"],
            gap=rec["heldout_lpd_best_minus_gt"], gap_se=rec["heldout_se"], recovered=rec["recovered"],
            threshold=rec["rmse_threshold"],
            best_seed=notes.get("best_seed"), best_seed_rmse=notes.get("best_seed_rmse_to_gt"),
        )
    best_seed = max((k for k in held if k.startswith("seed:")), key=lambda k: held[k]["lpd"])
    return dict(
        cell=cell_dir.name, condition=cell["condition"], replicate=cell["replicate"],
        ground_truth=cell.get("ground_truth"), excluded=cell.get("excluded_seeds", []),
        exported=best, n_live=len(live), counts=counts,
        best_seed=best_seed.split(":", 1)[1], best_seed_by_source=held[best_seed]["by_source"],
        exported_heldout=_held_for(best, held, live), n_heldout_models=len(held),
        exported_rank=1 + sorted((v["lpd"] for v in held.values()), reverse=True).index(_held_for(best, held, live)["lpd"]),
        models=models, recovery=recovery,
        cost=usage["cost_usd"], agent_runs=usage["n_calls"], tokens=usage["total_tokens"],
        wall=notes.get("wall"), max_rss_gb=notes.get("max_rss_gb"),
        report=f"cells/{cell_dir.name}.html",
    )


def build(sweep: Path) -> dict:
    sweep = Path(sweep)
    notes = json.loads((sweep / "run_notes.json").read_text())
    cells = sorted(p for p in sweep.iterdir() if (p / "cell.json").exists())
    if not cells:
        raise FileNotFoundError(f"no cells (<cell>/cell.json) in {sweep}")
    order = {"real": 0, "recovery_literal": 1, "recovery_salience": 2}
    bundles = [cell_bundle(c, notes["cells"].get(c.name, {})) for c in cells]
    bundles.sort(key=lambda b: (order.get(b["condition"], 9), b["replicate"]))
    return dict(sweep=sweep.name, date=notes.get("date"), code=notes.get("code"), source=notes.get("source"),
                title=notes.get("title", sweep.name), summary=notes.get("summary", ""),
                findings=notes.get("findings", []), decisions=notes.get("decisions", []),
                links=notes.get("links", []), cells=bundles)


def render(bundle: dict, template: Path = TEMPLATE) -> str:
    blob = json.dumps(bundle).replace("</", "<\\/")
    title = bundle["title"].replace("&", "&amp;").replace("<", "&lt;")
    return template.read_text(encoding="utf-8").replace("__PAGE_TITLE__", title).replace("__BUNDLE_JSON__", blob)


@dataclass
class Args:
    sweep: Path
    """A sweep's brought-back outputs (e.g. data/rsa/sherlock_run1)."""


def main(args: Args) -> None:
    bundle = build(args.sweep)
    out = Path(args.sweep) / "overview.html"
    out.write_text(render(bundle), encoding="utf-8")
    print(f"wrote {out} ({len(bundle['cells'])} cells)")


if __name__ == "__main__":
    main(tyro.cli(Args))
