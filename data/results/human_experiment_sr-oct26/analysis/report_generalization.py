"""Split each winner's gain over the best seed into shared structure and person-level parameters.

From ``generalization_analysis.py``'s point estimates and two-way bootstrap
replicates, per analysis (``within`` run S; ``refit`` / ``fixed`` run S on run
T) and per experiment winner of S:

    structure     = pooled winner - best seed   (population-level only, both)
    person-level  = full winner - pooled winner (every person-level parameter)
    total         = full winner - best seed

The best seed is chosen on the source run's own data (its ``within`` scores),
never on the target. Intervals: the point estimate plus the 2.5 / 97.5
percentiles of the bootstrap replicates' deviation from the unit-weight
replicate (the replicates use a subsample of the posterior and z draws; the
deviation removes that subsample's level). ``share_structure`` is
structure / total per replicate. Also the winner-to-winner steps, pooled and
full: did a later experiment's winner improve on the earlier one?

In ``within``, every winner was selected by a loop that saw some of the scored
people (all of them, for experiment 3's): flagged ``selection_overlap``.

Writes ``results/generalization/decomposition.csv`` and
``decomposition.pdf`` (plotnine).

Usage:
    uv run python data/results/human_experiment_sr-oct26/analysis/report_generalization.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here

ANALYSIS_ROOT = here() / "data/results/human_experiment_sr-oct26/analysis"
sys.path.insert(0, str(ANALYSIS_ROOT))

from fit_transfer_models import SEEDS, WINNERS  # noqa: E402

ANALYSES = [
    ("within", 1, 1),
    ("within", 2, 2),
    ("refit", 1, 2),
    ("refit", 2, 1),
    ("fixed", 1, 2),
    ("fixed", 2, 1),
]


@dataclass
class Args:
    """Report the structure / person-level split with two-way bootstrap intervals."""

    in_dir: Path = ANALYSIS_ROOT / "results/generalization"


def load(in_dir: Path, analysis: str, source: int, target: int):
    tag = f"{analysis}_run{source}_on_run{target}"
    table = pd.read_csv(in_dir / f"{tag}.csv").set_index("model")
    boot = np.load(in_dir / f"{tag}_boot.npz")
    return table, boot


def interval(point: float, reps: np.ndarray, unit: float) -> tuple[float, float, float]:
    dev = reps - unit
    return float(dev.std()), point + float(np.percentile(dev, 2.5)), point + float(np.percentile(dev, 97.5))


def main(args: Args) -> None:
    rows = []
    for analysis, source, target in ANALYSES:
        tag = f"{analysis}_run{source}_on_run{target}"
        if not (args.in_dir / f"{tag}.csv").exists():
            print(f"[report] {tag}: not computed yet, skipped")
            continue
        table, boot = load(args.in_dir, analysis, source, target)
        own, _ = load(args.in_dir, "within", source, source)
        best_seed = max(SEEDS, key=lambda s: own.elpd[s])
        per_1000 = 1000 / table.n_trials.iloc[0]

        def diff(a: str, b: str):
            point = table.elpd[a] - table.elpd[b]
            reps = boot[a] - boot[b]
            unit = float(boot[f"{a}__unit"] - boot[f"{b}__unit"])
            return point, reps, unit

        winners = list(WINNERS[source])
        for e, winner in enumerate(winners, start=1):
            parts = {
                "structure": diff(f"{winner}__pooled", best_seed),
                "person-level": diff(winner, f"{winner}__pooled"),
                "total": diff(winner, best_seed),
            }
            if e > 1:
                parts["step, pooled"] = diff(f"{winner}__pooled", f"{winners[e - 2]}__pooled")
                parts["step, full"] = diff(winner, winners[e - 2])
            s_point, s_reps, s_unit = parts["structure"]
            t_point, t_reps, t_unit = parts["total"]
            share_reps = (s_point + s_reps - s_unit) / (t_point + t_reps - t_unit)
            for part, (point, reps, unit) in parts.items():
                se, lo, hi = interval(point, reps, unit)
                rows.append(
                    {
                        "analysis": analysis,
                        "source_run": source,
                        "target_run": target,
                        "winner_of_experiment": e,
                        "winner": winner,
                        "best_seed": best_seed,
                        "part": part,
                        "nats": point,
                        "se": se,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_trials": int(table.n_trials.iloc[0]),
                        "per_1000_trials": point * per_1000,
                        "share_structure": s_point / t_point if part == "structure" else np.nan,
                        "share_ci_low": float(np.percentile(share_reps, 2.5)) if part == "structure" else np.nan,
                        "share_ci_high": float(np.percentile(share_reps, 97.5)) if part == "structure" else np.nan,
                        "selection_overlap": analysis == "within",
                    }
                )
    if not rows:
        raise FileNotFoundError(f"no generalization results in {args.in_dir}")
    out = pd.DataFrame(rows)
    out.to_csv(args.in_dir / "decomposition.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(
            out[["analysis", "source_run", "target_run", "winner_of_experiment", "part", "nats", "ci_low", "ci_high",
                 "per_1000_trials", "share_structure", "share_ci_low", "share_ci_high"]]
            .to_string(index=False, float_format="%.2f")
        )
    plot(out[out.part.isin(["structure", "person-level"])], args.in_dir / "decomposition.pdf")


def plot(df: pd.DataFrame, path: Path) -> None:
    from plotnine import (
        aes, element_text, facet_grid, geom_hline, geom_pointrange, ggplot, labs, position_dodge, theme, theme_bw,
    )

    scale = 1000 / df.n_trials
    df = df.assign(
        y=df.per_1000_trials,
        lo=df.ci_low * scale,
        hi=df.ci_high * scale,
        winner_label="exp " + df.winner_of_experiment.astype(str),
        panel=df.analysis + ": run " + df.source_run.astype(str) + " → run " + df.target_run.astype(str),
    )
    figure = (
        ggplot(df, aes("winner_label", "y", ymin="lo", ymax="hi", colour="part"))
        + geom_hline(yintercept=0, linetype="dashed", colour="grey")
        + geom_pointrange(position=position_dodge(width=0.5))
        + facet_grid("source_run ~ analysis", labeller="label_both")
        + labs(x="winner of experiment", y="gain (nats per 1,000 trials)", colour="")
        + theme_bw()
        + theme(legend_position="bottom", strip_text=element_text(size=8))
    )
    figure.save(path, width=9, height=5.5, verbose=False)


if __name__ == "__main__":
    main(tyro.cli(Args))
