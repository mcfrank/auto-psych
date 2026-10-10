"""Figure 3 for the sr-oct26 human runs: seeds vs. the best agent model, per run.

For each run, the four seed models and the run's experiment-3 winner, fitted on
all of that run's data (experiment 3's cumulative responses.csv). For runs 1
and 2 the fits are the generalization analysis's within-run fits
(fit_transfer_models.py, fits/transfer/run<R>_on_run<R>), each at the sampler
seed its record in results/generalization/fit_seeds says converged; for run 3
they are fit_run3_figure3.py's (fits/run3_experiment3, seeds in its
fit_seeds.json), made the same way. They are read from mcmc_cache only: a fit
missing from the cache raises instead of sampling.

Each model is plotted at its ELPD-LOO relative to the best agent model ± dse
(az.compare's trial-level SE of the difference, as in the paper's Figure 3).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

ANALYSIS_ROOT = here() / "data/results/human_experiment_sr-oct26/analysis"
FIT_SEEDS = ANALYSIS_ROOT / "results/generalization/fit_seeds"
SEEDS = ("falk_konold_dp", "finite_experience_occurrence", "local_representativeness", "motif_stack")
SEED_LABELS = {
    "falk_konold_dp": "encoding difficulty",
    "finite_experience_occurrence": "finite experience",
    "local_representativeness": "local representativeness",
    "motif_stack": "regularity detection",
}
# Each run's experiment-3 winner (collected/SUMMARY.md).
BEST_AGENT_MODEL = {
    1: "personal_pattern_detection_gain",
    2: "edge_streak_primacy_recency_pattern_rule",
    3: "mild_symmetric_trick_coin_streak_weary",
}
RUN3_FITS = ANALYSIS_ROOT / "fits/run3_experiment3"
BEST_AGENT = "best agent model"
# The generalization fits' sampler settings (fit_transfer_models.py Fit).
SAMPLER = dict(draws=3000, tune=2000, chains=4, target_accept=0.8)
# Fits whose recorded seed's fit is not in the local cache, keyed (model, run):
# another seed that converged on the same data here (logs/seed_search_edge_streak_run2_exp3.log).
SEED_OVERRIDES = {("edge_streak_primacy_recency_pattern_rule", 2): 2}


@dataclass
class Args:
    out_dir: Path = ANALYSIS_ROOT / "results"
    stem: str = "human_fit_comparison_experiment3_runs"


def fit_source(name: str, run: int) -> tuple[Path, int, float | None]:
    """(models dir, converged sampler seed, recorded ELPD-LOO or None) of a fit."""
    if run == 3:
        record = json.loads((RUN3_FITS / "fit_seeds.json").read_text(encoding="utf-8"))[name]
        if record["seed"] is None:
            raise ValueError(f"{name}@run3: no converged seed recorded")
        return RUN3_FITS / "models", record["seed"], None
    models_dir = ANALYSIS_ROOT / f"fits/transfer/run{run}_on_run{run}/models"
    record = json.loads((FIT_SEEDS / f"{name}@run{run}.json").read_text(encoding="utf-8"))
    if Path(record["models_dir"]) != models_dir.relative_to(here()):
        raise ValueError(f"{name}@run{run}: fitted from {record['models_dir']}, not {models_dir}")
    return models_dir, SEED_OVERRIDES.get((name, run), record["seed"]), record["elpd_loo"]


def run_rows(run: int) -> list[dict]:
    import arviz as az

    from src.models.clustered_se import cluster_dse, stimulus_clusters
    from src.models.pymc_inference import (
        cached_fit_path,
        clear_fit_cache,
        convergence_problems_of,
        fit_fingerprint,
        fit_model,
        resolve_fit_settings,
    )

    responses = (
        here()
        / f"data/results/human_experiment_sr-oct26/collected/run{run}/subjective_randomness"
        / "experiment3/model_loop/responses.csv"
    )
    names = [BEST_AGENT_MODEL[run], *SEEDS]
    loo, info = {}, {}
    for name in names:
        models_dir, seed, recorded = fit_source(name, run)
        kwargs = {**SAMPLER, "random_seed": seed}
        settings = resolve_fit_settings(name, models_dir, kwargs)
        nc = cached_fit_path(ANALYSIS_ROOT / "mcmc_cache", name, fit_fingerprint(name, models_dir, responses, settings))
        if not nc.exists():
            raise FileNotFoundError(f"{name}@run{run}: no cached fit {nc}; refusing to sample")
        fitted = fit_model(name, models_dir, responses, cache_dir=ANALYSIS_ROOT / "mcmc_cache", **kwargs)
        problems = convergence_problems_of(fitted)
        if problems:
            raise ValueError(f"{name}@run{run}: not converged: {problems}")
        diag = fitted.loo_diagnostics()
        loo[name] = diag.loo
        info[name] = {"loo_unreliable": diag.unreliable, "random_seed": seed}
        # The cached fit can be another sampling of the same (model, data, seed)
        # than the record's (e.g. fit_run_models.py's, on another machine).
        vs = "" if recorded is None else f"; record {recorded:.2f}, diff {float(diag.loo.elpd_loo) - recorded:+.2f}"
        print(f"[run {run}] {name}: ELPD-LOO {float(diag.loo.elpd_loo):.2f} (seed {seed}{vs})", flush=True)
        del fitted
        clear_fit_cache()

    cmp = az.compare(loo, ic="loo")
    best = cmp.index[0]
    if best != BEST_AGENT_MODEL[run]:
        raise ValueError(f"run {run}: rank 0 is {best!r}, not the best agent model; dse would not be against it")
    import csv

    with responses.open(encoding="utf-8", newline="") as f:
        clusters = stimulus_clusters(list(csv.DictReader(f)))
    rows = []
    for name, row in cmp.iterrows():
        rel = -float(row["elpd_diff"])
        dse = float(row["dse"])
        rows.append(
            {
                "run": f"run {run}",
                "model": name,
                "model_label": BEST_AGENT if name == best else SEED_LABELS[name],
                "kind": BEST_AGENT if name == best else "seed model",
                "elpd_loo": float(row["elpd_loo"]),
                "elpd_rel": rel,
                "dse": dse,
                "dse_clustered": 0.0
                if name == best
                else cluster_dse(np.asarray(loo[best].loo_i), np.asarray(loo[name].loo_i), clusters),
                "xmin": rel - dse,
                "xmax": rel + dse,
                **info[name],
            }
        )
    return rows


def forest_plot(df: pd.DataFrame, path: Path) -> None:
    from plotnine import (
        aes,
        element_text,
        facet_wrap,
        geom_errorbarh,
        geom_point,
        geom_vline,
        ggplot,
        labs,
        scale_color_manual,
        theme,
        theme_minimal,
    )

    df = df.copy()
    order = [BEST_AGENT, *(SEED_LABELS[s] for s in SEEDS)]
    df["model_label"] = pd.Categorical(df.model_label, categories=order[::-1], ordered=True)
    n = df.run.nunique()
    plot = (
        ggplot(df, aes(x="elpd_rel", y="model_label", color="kind"))
        + geom_vline(xintercept=0.0, linetype="dotted", color="grey", size=0.5)
        + geom_errorbarh(aes(xmin="xmin", xmax="xmax"), height=0.3, size=0.6)
        + geom_point(size=2.4)
        + facet_wrap("run", nrow=1)
        + scale_color_manual(values={BEST_AGENT: "#1B9E77", "seed model": "#7570B3"}, name="")
        + labs(x="ELPD-LOO relative to best agent model (± dse)", y="")
        + theme_minimal()
        + theme(
            figure_size=(3.6 * n, 3.6),
            legend_position="none",
            axis_title=element_text(size=17),
            axis_text=element_text(size=15),
            strip_text=element_text(size=17),
            panel_spacing=0.04,
        )
    )
    plot.save(path, dpi=150, verbose=False, bbox_inches="tight")


def main(args: Args) -> None:
    rows = [row for run in sorted(BEST_AGENT_MODEL) for row in run_rows(run)]
    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.stem}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
    pdf_path = args.out_dir / f"{args.stem}.pdf"
    forest_plot(df, pdf_path)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
