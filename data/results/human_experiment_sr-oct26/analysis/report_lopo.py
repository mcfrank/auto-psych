"""Leave-one-participant-out ELPD beside the loop's trial-level LOO, per run.

Reads ``lopo_analysis.py``'s per-person scores and the merged all-data tables
of ``compare_person_level.py --scheme all`` (their trial-level PSIS-LOO), and
reports per run:

* every model's LOPO ELPD and trial LOO, each as a difference from the best
  seed, and the two rankings;
* the paired LOPO comparisons of ``compare_person_level.py`` (structure alone,
  matched personalisation, full, person-level part) for each experiment's
  winner, and the winner-to-winner steps (did experiment 2's winner improve
  on experiment 1's, pooled and full?).

Paired SE over people: sqrt(n var(diff)). People share stimuli, so this
ignores the correlation through stimuli and is somewhat too small.

Writes ``results/lopo_models.csv`` and ``results/lopo_comparisons.csv``.

Usage:
    uv run python data/results/human_experiment_sr-oct26/analysis/report_lopo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here

RESULTS = here() / "data/results/human_experiment_sr-oct26/analysis/results"
SEEDS = ("falk_konold_dp", "finite_experience_occurrence", "local_representativeness", "motif_stack")
WINNERS = {
    1: ("individual_streak_aversion_lapse", "most_lopsided_stretch_aversion", "personal_pattern_detection_gain"),
    2: (
        "graded_periodicity_personal_ideal",
        "tally_span_switch_ideal_periodic_unit",
        "edge_streak_primacy_recency_pattern_rule",
    ),
}
# The winner variant nearest the personalised seeds' sensitivity + lapse.
MATCHED = {
    "individual_streak_aversion_lapse": "lapse_only",
    "most_lopsided_stretch_aversion": "sens_lapse",
    "personal_pattern_detection_gain": "sens_lapse",
    "graded_periodicity_personal_ideal": "sens",
    "tally_span_switch_ideal_periodic_unit": "sens",
    "edge_streak_primacy_recency_pattern_rule": "sens",
}


@dataclass
class Args:
    """Report LOPO ELPD beside trial-level LOO."""

    runs: tuple[int, ...] = (1, 2)


def main(args: Args) -> None:
    models, comparisons = [], []
    for run in args.runs:
        lopo = pd.read_csv(RESULTS / f"lopo_all_run{run}.csv").set_index("model")
        stuck = lopo[lopo.convergence_problems.notna() & (lopo.convergence_problems != "")]
        if len(stuck):
            raise ValueError(f"run {run}: LOPO scored unconverged fits: {list(stuck.index)}")
        people = pd.read_csv(RESULTS / f"lopo_all_people_run{run}.csv")
        wide = people.pivot(index="participant_id", columns="model", values="elpd_lopo")
        trial = pd.read_csv(RESULTS / f"person_level_all_merged_run{run}.csv").set_index("model")
        both = sorted(set(lopo.index) & set(trial.index))
        if set(lopo.index) - set(trial.index):
            raise ValueError(f"run {run}: no trial LOO for {sorted(set(lopo.index) - set(trial.index))}")
        if (lopo.random_seed[both] != trial.random_seed[both]).any():
            raise ValueError(f"run {run}: LOPO and trial LOO come from fits at different seeds")
        best_seed_lopo = max(SEEDS, key=lambda s: lopo.elpd_lopo[s])
        best_seed_trial = max(SEEDS, key=lambda s: trial.trial_loo_elpd[s])
        for name in both:
            models.append(
                {
                    "run": run,
                    "model": name,
                    "elpd_lopo": lopo.elpd_lopo[name],
                    "lopo_over_best_seed": lopo.elpd_lopo[name] - lopo.elpd_lopo[best_seed_lopo],
                    "trial_loo": trial.trial_loo_elpd[name],
                    "trial_over_best_seed": trial.trial_loo_elpd[name] - trial.trial_loo_elpd[best_seed_trial],
                    "max_pareto_k": lopo.max_pareto_k[name],
                    "n_person_vectors": lopo.n_person_vectors[name],
                }
            )

        personal_seed = max((f"{s}__personal" for s in SEEDS), key=lambda s: lopo.elpd_lopo[s])
        pairs = []
        for experiment, winner in enumerate(WINNERS[run], start=1):
            pairs += [
                ("structure alone", experiment, f"{winner}__pooled", best_seed_lopo),
                ("matched personalisation", experiment, f"{winner}__{MATCHED[winner]}", personal_seed),
                ("full", experiment, winner, personal_seed),
                ("person-level part", experiment, winner, f"{winner}__pooled"),
            ]
        for experiment in (2, 3):
            earlier, later = WINNERS[run][experiment - 2], WINNERS[run][experiment - 1]
            pairs += [
                ("winner step, pooled", experiment, f"{later}__pooled", f"{earlier}__pooled"),
                ("winner step, full", experiment, later, earlier),
            ]
        pairs.append(("seed personalisation", None, personal_seed, best_seed_lopo))
        for label, experiment, a, b in pairs:
            d = wide[a] - wide[b]
            comparisons.append(
                {
                    "run": run,
                    "comparison": label,
                    "experiment": experiment,
                    "model": a,
                    "versus": b,
                    "lopo_diff": float(d.sum()),
                    "lopo_dse": float(np.sqrt(len(d) * d.var())),
                    "people_better": int((d > 0).sum()),
                    "n_people": len(d),
                    "trial_loo_diff": float(trial.trial_loo_elpd[a] - trial.trial_loo_elpd[b]),
                }
            )

    models = pd.DataFrame(models).sort_values(["run", "elpd_lopo"], ascending=[True, False])
    models["lopo_rank"] = models.groupby("run").elpd_lopo.rank(ascending=False).astype(int)
    models["trial_rank"] = models.groupby("run").trial_loo.rank(ascending=False).astype(int)
    models.to_csv(RESULTS / "lopo_models.csv", index=False)
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(RESULTS / "lopo_comparisons.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 46):
        print(models.to_string(index=False, float_format="%.1f"))
        print()
        print(comparisons.drop(columns=["n_people"]).to_string(index=False, float_format="%.1f"))


if __name__ == "__main__":
    main(tyro.cli(Args))
