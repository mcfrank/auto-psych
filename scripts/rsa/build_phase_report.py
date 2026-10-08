"""The existing-data phase report: Sherlock runs 1 and 2, before the live experiments.

    uv run python -m scripts.rsa.build_phase_report

Reads the brought-back sweeps (data/rsa/sherlock_run1, sherlock_run2: the
same files the sweep overviews read, plus run 2's heldout/unit_lpd.csv and
recovery/recovery.json) and fills the numbers into
docs/auto_rsa/phase_report_template.html, writing
data/rsa/existing_data_report.html. The prose lives in the template; every
number on the page's charts and tables comes from here.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from src.rsa.sweep_report import build
from src.runtime.config import REPO_ROOT

RSA = REPO_ROOT / "data" / "rsa"
TEMPLATE = REPO_ROOT / "docs" / "auto_rsa" / "phase_report_template.html"
OUT = RSA / "existing_data_report.html"


def _first_sentence(path: Path) -> str:
    doc = " ".join((ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or "").split())
    return doc.split(". ")[0].rstrip(".") + "." if doc else ""


def claim1(sweeps: dict) -> list:
    rows = []
    for run, b in sweeps.items():
        for c in b["cells"]:
            if c["recovery"] is None:
                e = c["exported_heldout"]
                rows.append(dict(run=run, cell=c["cell"], exported=c["exported"], diff=e["diff"], se=e["se"],
                                 rank=c["exported_rank"], n=c["n_heldout_models"], best_seed=c["best_seed"],
                                 by_source={k: v - c["best_seed_by_source"][k] for k, v in e["by_source"].items()},
                                 hypothesis=_first_sentence(RSA / b["sweep"] / c["cell"] / "best_model.py")))
    return rows


def per_source(sweep: Path) -> list:
    """Exported minus the best seed per source, grouped CV and held out (run 2's unit_lpd)."""
    rows = []
    for cell in sorted(sweep.glob("real_rep*")):
        u = pd.read_csv(cell / "heldout" / "unit_lpd.csv")
        seed = next(m for m in u.model.unique() if m.endswith("(seed)"))
        exp = next(m for m in u.model.unique() if m.endswith("(exported)"))
        w = u.pivot_table(index=["split", "source"], columns="model", values="lpd", aggfunc="sum")
        d = (w[exp] - w[seed])
        e6 = u[(u.experiment == "E6_valence") & (u.split == "test")].pivot_table(index="model", values="lpd", aggfunc="sum").lpd
        rows.append(dict(cell=cell.name, cv=d.loc["cv"].to_dict(), test=d.loc["test"].to_dict(),
                         e6=float(e6[exp] - e6[seed])))
    return rows


def recovery(sweeps: dict) -> list:
    rows = []
    for run, b in sweeps.items():
        for c in b["cells"]:
            if c["recovery"] is None:
                continue
            cell = RSA / b["sweep"] / c["cell"]
            rec = json.loads((cell / "recovery" / "recovery.json").read_text())
            live = rec["live_pool_rmse_to_gt"]
            closest = min(live, key=live.get)
            rows.append(dict(
                run=run, cell=c["cell"], ground_truth=c["recovery"]["ground_truth"], exported=rec["best_model"],
                exported_rmse=rec["best_pool_rmse_to_gt"], closest=closest, closest_rmse=live[closest],
                best_seed=rec.get("best_seed"), best_seed_rmse=rec.get("best_seed_rmse_to_gt") or c["recovery"]["best_seed_rmse"],
                n_live=len(live), n_within=sum(v <= rec["rmse_threshold"] for v in live.values()),
                gap=rec["heldout_lpd_best_minus_gt"], gap_se=rec["heldout_se"], recovered=rec["recovered"],
                heldout=c["exported_heldout"]["diff"], hypothesis=_first_sentence(cell / "best_model.py"),
                rounds=[r["heldout"] for r in c["rounds"] if r["heldout"] is not None and r["round"] < c["n_rounds"]],
            ))
    return rows


def ops(sweeps: dict) -> list:
    return [dict(run=run, cell=c["cell"], cost=c["cost"], agent_runs=c["agent_runs"], wall=c["wall"],
                 max_rss_gb=c["max_rss_gb"], counts=c["counts"], n_live=c["n_live"]) for run, b in sweeps.items()
            for c in b["cells"]]


def main() -> None:
    sweeps = {"run 1": build(RSA / "sherlock_run1"), "run 2": build(RSA / "sherlock_run2")}
    run2 = sweeps["run 2"]
    data = dict(
        claim1=claim1(sweeps), per_source=per_source(RSA / "sherlock_run2"), recovery=recovery(sweeps), ops=ops(sweeps),
        rounds=[dict(cell=c["cell"], replicate=c["replicate"], n_rounds=c["n_rounds"],
                     steps=[r for r in c["rounds"] if r["round"] < c["n_rounds"]])
                for c in run2["cells"] if c["recovery"] is None],
    )
    html = TEMPLATE.read_text(encoding="utf-8")
    if "__DATA_JSON__" not in html:
        raise ValueError(f"{TEMPLATE} lacks the __DATA_JSON__ placeholder")
    OUT.write_text(html.replace("__DATA_JSON__", json.dumps(data).replace("</", "<\\/")), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
