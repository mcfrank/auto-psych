"""Model-vs-people report pages for the RSA domain.

    uv run python -m src.rsa.report --comparison-dir data/rsa/seed_comparison_all \
        --out-html data/rsa/report.html

Builds a JSON *bundle* from a comparison directory (`src.rsa.compare_seeds`
output: comparison.csv, params.csv, cells.csv, summary.json) and renders it
into one self-contained HTML page (`report_template.html`, inline SVG, no
network). The page has three levels:

* models: PSIS-LOO standing (ELPD difference from the best, +/- its SE),
  Pearson r and RMSE over cells, posterior parameter intervals, hypothesis;
* fit by experiment: RMSE of each model in each experiment (heatmap);
* predictions vs people: one small-multiple panel per display cell (game x
  word x condition), people's choice proportions (95% intervals) against the
  selected models' predictions, with the game drawn under the bars.

The bundle is the stable interface: loop stages can write the same shape
(one entry per experiment of a run) and the page renders it unchanged.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tyro

from src.models.model_manifest import read_manifest_entries
from src.runtime.config import PROJECT_ASSETS_DIR

TEMPLATE = Path(__file__).with_name("report_template.html")
SEED_DIR = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
BUNDLE_VERSION = 1

# What each pragmods experiment manipulated, for panel headers.
EXPERIMENT_LABELS = {
    "E1_dv": "E1 · response format",
    "E2_manip_check": "E2 · manipulation check",
    "E3_ling_frame": "E3 · one word vs sentence",
    "E4_prior_frame": "E4 · prior question wording",
    "E5_baserate": "E5 · familiarization base rate",
    "E6_valence": "E6 · favorite vs least favorite",
    "E7_color": "E7 · color salience",
    "color_prior_rerun": "E7b · color prior rerun",
    "E8_levels": "E8 · levels of inference",
    "levels_prior_action": "E8b · levels prior (next action)",
    "E9_twins": "E9 · twins",
    "E10_oddman": "E10 · odd one out",
    "size": "Size · 2–4 objects × 2–4 features",
    "sequences": "Sequences · repeated trials",
    "speakers": "Speakers · listener trials",
}


@dataclass
class Args:
    comparison_dir: Path
    out_html: Path
    models_dir: Path = SEED_DIR
    title: str = "Pragmods seed models"
    dataset_label: str = "pragmods forced-choice trials"
    min_cell_n: int = 10  # cells with fewer trials stay in the panels but not in RMSE/r
    write_bundle: bool = True  # also write <out_html>.bundle.json


def _rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def _finite(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def build_bundle(comparison_dir: Path, models_dir: Path, *, title: str,
                 dataset_label: str, min_cell_n: int) -> Dict[str, Any]:
    comparison_dir = Path(comparison_dir)
    comp = pd.read_csv(comparison_dir / "comparison.csv", index_col=0)
    params = pd.read_csv(comparison_dir / "params.csv")
    cells = pd.read_csv(comparison_dir / "cells.csv")
    summary = json.loads((comparison_dir / "summary.json").read_text())
    names = list(comp.index)
    missing = [n for n in names if n not in cells.columns]
    if missing:
        raise ValueError(f"cells.csv has no prediction column for {missing}")
    rationale = {e["name"]: e.get("rationale", "") for e in read_manifest_entries(models_dir)}

    big = cells[cells["n"] >= min_cell_n]
    experiments = [e for e in EXPERIMENT_LABELS if e in set(cells["experiment"])]
    experiments += sorted(set(cells["experiment"]) - set(experiments))

    models = []
    for name in names:
        row = comp.loc[name]
        problems = row.get("convergence_problems")
        problems = "" if isinstance(problems, float) else str(problems)
        by_exp = {
            e: _finite(_rmse(g["observed"], g[name]))
            for e, g in big.groupby("experiment")
        }
        models.append(
            dict(
                name=name,
                rationale=rationale.get(name, ""),
                rank=int(row["rank"]),
                elpd_loo=float(row["elpd_loo"]),
                se=float(row["se"]),
                elpd_diff=float(row["elpd_diff"]),
                dse=float(row["dse"]),
                p_loo=float(row["p_loo"]),
                loo_reliable=bool(row["loo_reliable"]),
                converged=not problems,
                convergence_problems=problems,
                r=_finite(float(np.corrcoef(big["observed"], big[name])[0, 1])),
                rmse=_finite(_rmse(big["observed"], big[name])),
                rmse_by_experiment=by_exp,
                params=[
                    dict(param=p["param"], mean=p["mean"], lo=p["lo"], hi=p["hi"])
                    for _, p in params[params["model"] == name].iterrows()
                ],
            )
        )
    models.sort(key=lambda m: m["rank"])

    keys = ["experiment", "condition", "objects", "query", "utterance"]
    panels = []
    for key, g in cells.groupby(keys, sort=False):
        d = dict(zip(keys, key))
        objects = json.loads(d["objects"])
        classes = []
        for _, r in g.sort_values("object").iterrows():
            obj = int(r["object"])
            classes.append(
                dict(
                    object=obj,
                    copies=sum(1 for o in objects if o == objects[obj]),
                    count=int(r["count"]),
                    observed=float(r["observed"]),
                    preds={m: float(r[m]) for m in names},
                )
            )
        panels.append(
            dict(
                experiment=d["experiment"],
                condition=str(d["condition"]),
                objects=objects,
                query=d["query"],
                utterance=None if int(d["utterance"]) < 0 else int(d["utterance"]),
                n=int(g["n"].iloc[0]),
                classes=classes,
            )
        )
    order = {e: i for i, e in enumerate(experiments)}
    panels.sort(key=lambda p: (order[p["experiment"]], p["query"] != "utterance", p["condition"]))

    return dict(
        version=BUNDLE_VERSION,
        title=title,
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        dataset=dict(
            label=dataset_label,
            n_trials=summary.get("n_trials"),
            settings=summary.get("settings"),
            min_cell_n=min_cell_n,
            n_cells_scored=int(big.groupby(keys).ngroups),
        ),
        experiments=[dict(id=e, label=EXPERIMENT_LABELS.get(e, e)) for e in experiments],
        models=models,
        panels=panels,
    )


def render(bundle: Dict[str, Any], template: Path = TEMPLATE) -> str:
    html = template.read_text(encoding="utf-8")
    payload = json.dumps(bundle, allow_nan=False).replace("</", "<\\/")
    if "__BUNDLE_JSON__" not in html or "__PAGE_TITLE__" not in html:
        raise ValueError(f"{template} lacks the __BUNDLE_JSON__ / __PAGE_TITLE__ placeholders")
    title = bundle["title"].replace("&", "&amp;").replace("<", "&lt;")
    return html.replace("__PAGE_TITLE__", title).replace("__BUNDLE_JSON__", payload)


def main(args: Args) -> None:
    bundle = build_bundle(
        args.comparison_dir, args.models_dir, title=args.title,
        dataset_label=args.dataset_label, min_cell_n=args.min_cell_n,
    )
    out = Path(args.out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(bundle), encoding="utf-8")
    if args.write_bundle:
        out.with_suffix(".bundle.json").write_text(json.dumps(bundle, indent=1, allow_nan=False))
    print(f"wrote {out} ({len(bundle['panels'])} panels, {len(bundle['models'])} models)")


if __name__ == "__main__":
    main(tyro.cli(Args))
