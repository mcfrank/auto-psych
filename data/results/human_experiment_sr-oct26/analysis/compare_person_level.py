"""Merge the person-level results with their rescored rows and compare the runs.

``person_level_analysis.py`` writes a full table per (run, scheme) and, for
models refit at another sampler seed (``--only``; the seed search in
``find_converged_seed.py``), extra tables of the same people's scores. This
replaces each rescored model's rows with the rescored ones, refuses a merged
model whose fit still has convergence problems, and reports, per run, the
paired new-participant comparisons that answer "is the discovered structure
better beyond the person-level parameters?":

  structure alone          pooled winner vs the best (unpersonalised) seed
  matched personalisation  winner keeping its sensitivity (+ lapse) vs the
                           best personalised seed
  full                     full winner vs the best personalised seed
  person-level part        full winner vs its pooled version

for the winners of experiments 1 and 2, the clean ones on held-out people
(selected before experiment 3's people existed). Paired SE over people:
sqrt(n var(diff)).

Writes ``results/person_level_<scheme>_merged_run<N>.csv`` and
``results/person_level_<scheme>_comparisons.csv``.

Usage:
    uv run python data/results/human_experiment_sr-oct26/analysis/compare_person_level.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here

RESULTS = here() / "data/results/human_experiment_sr-oct26/analysis/results"
SEEDS = ("falk_konold_dp", "finite_experience_occurrence", "local_representativeness", "motif_stack")
CLEAN_WINNERS = {
    1: ("individual_streak_aversion_lapse", "most_lopsided_stretch_aversion"),
    2: ("graded_periodicity_personal_ideal", "tally_span_switch_ideal_periodic_unit"),
}
# The winner variant nearest the personalised seeds' sensitivity + lapse: the
# winner's own person-level ones, where it has them (run 1's experiment-1
# winner has a shared sensitivity; run 2's winners no lapse).
MATCHED = {
    "individual_streak_aversion_lapse": "lapse_only",
    "most_lopsided_stretch_aversion": "sens_lapse",
    "graded_periodicity_personal_ideal": "sens",
    "tally_span_switch_ideal_periodic_unit": "sens",
}
# Rescored tables per (run, scheme), in the order they were written.
RESCORES = {
    (1, "heldout"): ["only"],
    (1, "all"): ["only"],
    (2, "heldout"): ["rescore1", "only"],
    (2, "all"): ["rescore1", "only"],
}


@dataclass
class Args:
    """Merge rescored person-level rows and compare the runs."""

    scheme: Literal["heldout", "all"] = "heldout"
    runs: tuple[int, ...] = (1, 2)


def table_path(run: int, scheme: str, tag: str, people: bool) -> Path:
    stem = f"person_level_{scheme}" + ("_people" if people else "")
    suffix = {"": "", "only": "_only"}.get(tag, f"_{tag}")
    return RESULTS / f"{stem}{suffix}{'' if run == 1 else f'_run{run}'}.csv"


def merged(run: int, scheme: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(table_path(run, scheme, "", False))
    people = pd.read_csv(table_path(run, scheme, "", True))
    for tag in RESCORES[(run, scheme)]:
        path = table_path(run, scheme, tag, False)
        if not path.exists():
            raise FileNotFoundError(f"rescored table missing: {path}")
        extra = pd.read_csv(path)
        extra_people = pd.read_csv(table_path(run, scheme, tag, True))
        table = pd.concat([table[~table.model.isin(extra.model)], extra])
        people = pd.concat([people[~people.model.isin(extra.model)], extra_people])
    table = table.drop(columns=[c for c in table.columns if c.startswith("new_participant_d")])
    return table.sort_values("new_participant_score", ascending=False), people


def main(args: Args) -> None:
    rows = []
    for run in args.runs:
        table, people = merged(run, args.scheme)
        stuck = table[table.convergence_problems.notna() & (table.convergence_problems != "")]
        wide = people.pivot(index="participant_id", columns="model", values="score")
        score = table.set_index("model").new_participant_score
        r2 = table.set_index("model").pop_r2_stimulus
        out = RESULTS / f"person_level_{args.scheme}_merged_run{run}.csv"
        table.to_csv(out, index=False)

        def best(names):
            names = [n for n in names if n in score.index]
            return max(names, key=score.get)

        seed = best(SEEDS)
        personal_seed = best([f"{s}__personal" for s in SEEDS])
        comparisons = []
        for experiment, winner in enumerate(CLEAN_WINNERS[run], start=1):
            comparisons += [
                ("structure alone", experiment, f"{winner}__pooled", seed),
                ("matched personalisation", experiment, f"{winner}__{MATCHED[winner]}", personal_seed),
                ("full", experiment, winner, personal_seed),
                ("person-level part", experiment, winner, f"{winner}__pooled"),
            ]
        comparisons.append(("seed personalisation", None, personal_seed, seed))
        for label, experiment, a, b in comparisons:
            missing = {a, b} - set(wide.columns)
            if missing:
                raise ValueError(f"run {run}: no scores for {missing}")
            d = wide[a] - wide[b]
            rows.append(
                {
                    "run": run,
                    "comparison": label,
                    "winner_of_experiment": experiment,
                    "model": a,
                    "versus": b,
                    "diff": float(d.sum()),
                    "dse": float(np.sqrt(len(d) * d.var())),
                    "people_better": int((d > 0).sum()),
                    "n_people": len(d),
                    "pop_r2_model": float(r2[a]),
                    "pop_r2_versus": float(r2[b]),
                    "unconverged_in_comparison": ", ".join(sorted({a, b} & set(stuck.model))),
                }
            )
        if len(stuck):
            print(f"run {run}: fits with convergence problems: {', '.join(stuck.model)}")
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / f"person_level_{args.scheme}_comparisons.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 48):
        print(result.drop(columns=["n_people"]).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main(tyro.cli(Args))
