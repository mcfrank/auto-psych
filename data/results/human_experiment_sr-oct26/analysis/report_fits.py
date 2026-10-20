"""Seed-vs-discovered model comparison for sr-oct26 run 1, as in the paper.

Reads what ``fit_run_models.py`` wrote (no MCMC) and produces the paper's
model-comparison analyses for this run:

* ``human_fit_comparison.pdf``/``.csv`` (and ``_experiment3``): the forest
  plot of each seed's ELPD-LOO relative to the best discovered model, ± dse
  (the paper's Figure; ``plot_human_fit_comparison.py``, removed in 8d29ad1).
* ``model_comparison_experiment3.csv``/``.tex``: the paper's Table 1 on all
  of run 1's data (7,680 trials): k, ELPD-LOO, ΔELPD, SE(Δ) — and the
  stimulus-clustered SE the loop prunes by — and RMSE / R² over per-stimulus
  choice proportions.
* ``noise_ceiling.json``: split-half (Spearman–Brown) and variance-
  decomposition reliability of the per-stimulus proportions.
* ``model_rmse_experiment3_{matrix,pairs}.csv``: pairwise RMSE between the
  discovered models' population-level predictions (the paper's 0.092 / 0.029).

A stimulus is the unordered pair (left/right is randomized per person in this
series; the clustered SE uses the same unit), scored as the proportion
choosing the alphabetically first sequence. Model predictions are in-sample
posterior means per trial, averaged over a stimulus's trials; for person-level
models that is each person's own fitted prediction.

Usage:
    uv run python data/results/human_experiment_sr-oct26/analysis/report_fits.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here

RUN_ROOT = here() / "data/results/human_experiment_sr-oct26/collected/run1/subjective_randomness"
ANALYSIS_ROOT = here() / "data/results/human_experiment_sr-oct26/analysis"
SEEDS = ("falk_konold_dp", "finite_experience_occurrence", "local_representativeness", "motif_stack")
SEED_LABELS = {
    "falk_konold_dp": "Falk & Konold DP",
    "finite_experience_occurrence": "Finite experience",
    "local_representativeness": "Local representativeness",
    "motif_stack": "Motif stack",
}
BEST_AGENT = "best discovered model"
# Each experiment's winner (exported best) and the four tied at the end of experiment 3.
ROLES = {
    "individual_streak_aversion_lapse": "Win exp. 1",
    "most_lopsided_stretch_aversion": "Win exp. 2",
    "personal_pattern_detection_gain": "Win exp. 3",
    "capacity_blur_pattern_gain": "Tied exp. 3",
    "returns_to_balance_expectation": "Tied exp. 3",
    "sensitivity_linked_consistency": "Tied exp. 3",
}
N_SPLITS = 200


@dataclass
class Args:
    """Report the refits as the paper's figure, table and similarity numbers."""

    fits_dir: Path = ANALYSIS_ROOT / "fits"
    out_dir: Path = ANALYSIS_ROOT / "results"
    seed: int = 0


def load_experiment(fits_dir: Path, e: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    out = fits_dir / f"experiment{e}"
    compare = json.loads((out / "compare.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(out / "predictions.csv")
    n_params = pd.read_csv(out / "n_params.csv")
    return compare, preds, n_params


def forest_rows(e: int, compare: dict) -> list[dict]:
    """The seeds and the best non-seed model, relative to that model."""
    agents = [m for m in compare if m not in SEEDS]
    best = max(agents, key=lambda m: compare[m]["elpd_loo"])
    if compare[best]["rank"] != 0:
        raise ValueError(
            f"experiment {e}: best discovered model {best!r} is not rank 0, so the "
            "dse are not measured against it."
        )
    rows = []
    for m in (best, *SEEDS):
        c = compare[m]
        rel = -c["elpd_diff"]
        rows.append(
            {
                "experiment": f"experiment {e}",
                "model": m,
                "model_label": BEST_AGENT if m == best else SEED_LABELS[m],
                "kind": BEST_AGENT if m == best else "seed model",
                "elpd_loo": c["elpd_loo"],
                "elpd_rel": rel,
                "dse": c["dse"],
                "dse_clustered": c["dse_clustered"],
                "xmin": rel - c["dse"],
                "xmax": rel + c["dse"],
                "loo_unreliable": c["loo_unreliable"],
                "not_converged": c["not_converged"],
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
    n = df.experiment.nunique()
    plot = (
        ggplot(df, aes(x="elpd_rel", y="model_label", color="kind"))
        + geom_vline(xintercept=0.0, linetype="dotted", color="grey", size=0.5)
        + geom_errorbarh(aes(xmin="xmin", xmax="xmax"), height=0.3, size=0.6)
        + geom_point(size=2.4)
        + facet_wrap("experiment", nrow=1, scales="free_x")
        + scale_color_manual(values={BEST_AGENT: "#1B9E77", "seed model": "#7570B3"}, name="")
        + labs(x="ELPD-LOO relative to best discovered model (± dse)", y="")
        + theme_minimal()
        + theme(
            figure_size=(4.2 * n, 3.6),
            legend_position="none",
            axis_title=element_text(size=14),
            axis_text=element_text(size=12),
            strip_text=element_text(size=14),
        )
    )
    plot.save(path, dpi=150, verbose=False, bbox_inches="tight")


def stimulus_frame(responses: pd.DataFrame) -> pd.DataFrame:
    """Per trial: the stimulus key and whether the alphabetically first sequence was chosen."""
    first = np.minimum(responses.sequence_a.values, responses.sequence_b.values)
    a_is_first = responses.sequence_a.values == first
    return pd.DataFrame(
        {
            "stimulus": [" | ".join(sorted(p)) for p in zip(responses.sequence_a, responses.sequence_b)],
            "a_is_first": a_is_first,
            "chose_first": np.where(a_is_first, responses.chose_left, 1 - responses.chose_left),
        }
    )


def stimulus_fit(stim: pd.DataFrame, p_left: np.ndarray) -> tuple[float, float]:
    """RMSE and R² between observed and predicted per-stimulus proportions."""
    p_first = np.where(stim.a_is_first, p_left, 1 - p_left)
    agg = stim.assign(p=p_first).groupby("stimulus").agg(obs=("chose_first", "mean"), pred=("p", "mean"))
    resid = agg.obs - agg.pred
    r2 = 1 - (resid**2).sum() / ((agg.obs - agg.obs.mean()) ** 2).sum()
    return float(np.sqrt((resid**2).mean())), float(r2)


def noise_ceiling(stim: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Reliability of the per-stimulus proportions (02_best_fitting_theories.qmd)."""
    obs = stim.groupby("stimulus").chose_first.agg(["mean", "size"])
    noise = (obs["mean"] * (1 - obs["mean"]) / (obs["size"] - 1)).mean()
    analytic = 1 - noise / obs["mean"].var()
    rs = []
    for _ in range(N_SPLITS):
        half = stim.groupby("stimulus").chose_first.transform(
            lambda s: pd.Series(rng.permutation(np.arange(len(s)) % 2), index=s.index)
        )
        h = stim.assign(half=half).groupby(["stimulus", "half"]).chose_first.mean().unstack()
        rs.append(np.corrcoef(h[0], h[1])[0, 1])
    r_half = float(np.mean(rs))
    return {
        "n_stimuli": int(len(obs)),
        "responses_per_stimulus": [int(obs["size"].min()), int(obs["size"].max())],
        "split_half_r": r_half,
        "spearman_brown": 2 * r_half / (1 + r_half),
        "variance_decomposition": float(analytic),
    }


def pool_rmse(pool: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pool.pivot_table(index=["sequence_a", "sequence_b"], columns="model", values="p_left")[names]
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    pairs = []
    for i, a in enumerate(names):
        for b in names:
            ok = wide[a].notna() & wide[b].notna()
            mat.loc[a, b] = float(np.sqrt(((wide[a][ok] - wide[b][ok]) ** 2).mean()))
        for b in names[i + 1 :]:
            pairs.append({"model_a": a, "model_b": b, "rmse": mat.loc[a, b]})
    return mat, pd.DataFrame(pairs)


def latex_table(t: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"  \toprule",
        r"  Model & Type & $k$ & ELPD-LOO & $\Delta$ELPD & SE($\Delta$) & SE$_\mathrm{stim}$($\Delta$) & RMSE & $R^2$ \\",
        r"  \midrule",
    ]
    for _, r in t.iterrows():
        k = f"{r.n_population_params}" + (f"+{r.n_person_level_vectors}p" if r.n_person_level_vectors else "")
        name = r.model.replace("_", r"\_")
        lines.append(
            f"  {name} & {r.type} & {k} & {r.elpd_loo:.1f} & {r.elpd_diff:.1f} & {r.dse:.1f} & "
            f"{r.dse_clustered:.1f} & {r.rmse_stimulus:.3f} & {r.r2_stimulus:.3f} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    experiments = sorted(int(p.name.removeprefix("experiment")) for p in args.fits_dir.glob("experiment*") if (p / "compare.json").exists())
    if not experiments:
        raise FileNotFoundError(f"no compare.json under {args.fits_dir}; run fit_run_models.py first")

    # Forest plot: seeds vs best discovered model, per experiment.
    rows = []
    for e in experiments:
        compare, _, _ = load_experiment(args.fits_dir, e)
        rows += forest_rows(e, compare)
    forest = pd.DataFrame(rows)
    forest.to_csv(args.out_dir / "human_fit_comparison.csv", index=False)
    forest_plot(forest, args.out_dir / "human_fit_comparison.pdf")
    last = forest[forest.experiment == f"experiment {max(experiments)}"]
    last.to_csv(args.out_dir / f"human_fit_comparison_experiment{max(experiments)}.csv", index=False)
    forest_plot(last, args.out_dir / f"human_fit_comparison_experiment{max(experiments)}.pdf")

    if 3 not in experiments:
        print("experiment 3 not fitted yet: wrote the forest plot only")
        return

    # Table 1 on all of the run's data.
    compare, preds, n_params = load_experiment(args.fits_dir, 3)
    responses = pd.read_csv(RUN_ROOT / "experiment3/model_loop/responses.csv")
    stim = stimulus_frame(responses)
    table = []
    for m, c in compare.items():
        p = preds[preds.model == m].sort_values("row").p_left.to_numpy()
        if len(p) != len(responses):
            raise ValueError(f"{m}: {len(p)} predictions for {len(responses)} trials")
        rmse, r2 = stimulus_fit(stim, p)
        k = n_params.set_index("model").loc[m]
        table.append(
            {
                "model": m,
                "type": "Seed" if m in SEEDS else ROLES[m],
                "n_population_params": int(k.n_population_params),
                "n_person_level_vectors": int(k.n_person_level_vectors),
                "elpd_loo": c["elpd_loo"],
                "elpd_diff": c["elpd_diff"],
                "dse": c["dse"],
                "dse_clustered": c["dse_clustered"],
                "rmse_stimulus": rmse,
                "r2_stimulus": r2,
                "loo_unreliable": c["loo_unreliable"],
                "not_converged": c["not_converged"],
                "convergence_problems": "; ".join(c["convergence_problems"]),
            }
        )
    table = pd.DataFrame(table).sort_values("elpd_loo", ascending=False)
    table.to_csv(args.out_dir / "model_comparison_experiment3.csv", index=False)
    (args.out_dir / "model_comparison_experiment3.tex").write_text(latex_table(table), encoding="utf-8")

    ceiling = noise_ceiling(stim, np.random.default_rng(args.seed))
    best = table.iloc[0]
    ceiling["best_model"] = best.model
    ceiling["best_r2_stimulus"] = best.r2_stimulus
    ceiling["best_share_of_explainable_variance"] = best.r2_stimulus / ceiling["spearman_brown"]
    (args.out_dir / "noise_ceiling.json").write_text(json.dumps(ceiling, indent=2), encoding="utf-8")

    # Pairwise RMSE between the discovered models on the loop's novelty pool.
    pool = pd.read_csv(args.fits_dir / "experiment3/pool_predictions.csv")
    discovered = [m for m in table.model if m not in SEEDS]
    mat, pairs = pool_rmse(pool, discovered)
    mat.to_csv(args.out_dir / "model_rmse_experiment3_matrix.csv", float_format="%.6f")
    pairs.to_csv(args.out_dir / "model_rmse_experiment3_pairs.csv", index=False, float_format="%.6f")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(forest[["experiment", "model", "elpd_rel", "dse", "dse_clustered"]].to_string(index=False))
        print()
        print(table.drop(columns=["convergence_problems"]).to_string(index=False, float_format="%.3f"))
        print()
        print(json.dumps(ceiling, indent=2))
        print()
        print(mat.round(3).to_string())


if __name__ == "__main__":
    main(tyro.cli(Args))
