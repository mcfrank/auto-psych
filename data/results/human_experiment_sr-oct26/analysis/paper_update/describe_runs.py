"""describe_run.py's descriptives for all three sr-oct26 runs (no MCMC).

Reuses describe_run.py's own functions (pointing its RUN_ROOT at each run in
turn), so every definition matches the run-1 analysis: collection minutes from
the Prolific polls, H/T-consolidated unordered pairs for stimulus reuse, each
experiment's distinct sequences for lengths, and per-person proportions with
t-based 95% CIs for the biases (the unit is a person in one experiment).
"""

from __future__ import annotations

import importlib.util
import json
import sys

import pandas as pd
from pyprojroot import here

sys.path.insert(0, str(here()))
spec = importlib.util.spec_from_file_location(
    "describe_run", here() / "data/results/human_experiment_sr-oct26/analysis/describe_run.py"
)
dr = importlib.util.module_from_spec(spec)
sys.modules["describe_run"] = dr
spec.loader.exec_module(dr)

RUNS = (1, 2, 3)

frames, collection = [], []
for run in RUNS:
    dr.RUN_ROOT = here() / f"data/results/human_experiment_sr-oct26/collected/run{run}/subjective_randomness"
    t = dr.load_trials()
    t.insert(0, "run", run)
    t["participant_uid"] = f"r{run}_" + t.participant_uid
    frames.append(t)
    collection += [{"run": run, **dr.collection_minutes(e)} for e in dr.EXPERIMENTS]
trials = pd.concat(frames, ignore_index=True)

# Stimulus reuse across all nine experiments.
shown = trials.assign(
    unordered=lambda d: [" | ".join(sorted(p)) for p in zip(d.sequence_a, d.sequence_b)],
    pair=lambda d: [dr.canonical_pair(x, y) for x, y in zip(d.sequence_a, d.sequence_b)],
).drop_duplicates(["run", "experiment", "unordered"])
n_cells = shown.groupby("pair").apply(lambda g: g[["run", "experiment"]].drop_duplicates().shape[0])
raw_distinct = shown.unordered.nunique()

seqs = pd.concat(
    [trials[["run", "experiment", "sequence_a"]].rename(columns={"sequence_a": "s"}),
     trials[["run", "experiment", "sequence_b"]].rename(columns={"sequence_b": "s"})]
).drop_duplicates()
seqs["length"] = seqs.s.str.len()
lengths = seqs.groupby(["run", "experiment"]).length.agg(["count", "min", "median", "max"]).reset_index()
pair_len = shown.sequence_a.str.len()

bt = dr.bias_trials(trials)
bt["run"] = trials.run
per_person = bt.groupby(["run", "experiment", "participant_uid"])[list(dr.BIASES)].mean().reset_index()
topline = {b: dr.t_summary(per_person[b]) for b in dr.BIASES}
by_run = {int(r): {b: dr.t_summary(g[b]) for b in dr.BIASES} for r, g in per_person.groupby("run")}

coll = pd.DataFrame(collection)
out = {
    "n_trials": len(trials),
    "n_trials_by_run": {int(r): int(n) for r, n in trials.groupby("run").size().items()},
    "n_participant_experiments": int(trials.participant_uid.nunique()),
    "participants_per_experiment": {f"run{r}_exp{e}": int(g.participant_uid.nunique()) for (r, e), g in trials.groupby(["run", "experiment"])},
    "trials_per_participant": sorted(trials.groupby("participant_uid").size().unique().tolist()),
    "collection_minutes": collection,
    "collection_minutes_range": [float(coll.minutes.min()), float(coll.minutes.max())],
    "stimulus_pairs": {
        "pairs_shown": len(shown),
        "distinct_unordered_pairs": int(raw_distinct),
        "distinct_pairs_ht_consolidated": int(len(n_cells)),
        "fraction_distinct_ht_consolidated": len(n_cells) / len(shown),
        "fraction_distinct_unordered": raw_distinct / len(shown),
        "consolidated_pairs_in_more_than_one_experiment": int((n_cells > 1).sum()),
        "different_length_pairs": int((trials.sequence_a.str.len() != trials.sequence_b.str.len()).sum()),
        "pair_length_counts": {int(k): int(v) for k, v in pair_len.value_counts().sort_index().items()},
        "pair_length_median": float(pair_len.median()),
    },
    "sequence_length_medians_by_experiment": {f"run{int(r.run)}_exp{int(r.experiment)}": float(r["median"]) for _, r in lengths.iterrows()},
    "biases_topline": topline,
    "biases_by_run": by_run,
}
print(json.dumps(out, indent=2, default=float))
