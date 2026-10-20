"""Descriptive statistics of sr-oct26 run 1, as the paper reports them.

The counterpart of the September 2026 analysis in ``analysis/behavioral/``
(``00_load_human_data.qmd``, ``01_response_analysis.qmd``; removed in
ce3d200): collection time per experiment, stimulus-pair reuse across
experiments (H/T-consolidated), sequence lengths, and the four choice biases
(side, heads, length, alternation) with the participant as the unit and
t-based 95% CIs. No MCMC.

Every pair in this series is same-length (the design enumerates same-length
pairs only), so the length bias has no informative trial and is reported as
such.

Writes ``<out_dir>/descriptives.json``, ``biases_by_experiment.csv`` and
``biases_by_experiment.pdf``.

Usage:
    uv run python data/results/human_experiment_sr-oct26/analysis/describe_run.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here
from scipy import stats

RUN_ROOT = here() / "data/results/human_experiment_sr-oct26/collected/run1/subjective_randomness"
ANALYSIS_ROOT = here() / "data/results/human_experiment_sr-oct26/analysis"
EXPERIMENTS = (1, 2, 3)
BIASES = ("side", "n_heads", "length", "alternation")

_POLL = re.compile(r"^\[(\S+Z)\] Prolific poll: .* completed=(\d+) target=(\d+)")


@dataclass
class Args:
    """Describe run 1's data the way the paper describes the September runs."""

    out_dir: Path = ANALYSIS_ROOT / "results"


def complement(s: str) -> str:
    return s.translate(str.maketrans("HT", "TH"))


def canonical_pair(a: str, b: str) -> str:
    """The unordered pair, consolidated with its whole-display H/T relabelling."""
    raw = " | ".join(sorted((a, b)))
    comp = " | ".join(sorted((complement(a), complement(b))))
    return min(raw, comp)


def alternation_rate(s: str) -> float:
    return sum(x != y for x, y in zip(s, s[1:])) / (len(s) - 1)


def load_trials() -> pd.DataFrame:
    frames = []
    for e in EXPERIMENTS:
        d = pd.read_csv(RUN_ROOT / f"experiment{e}/data/responses.csv")
        d.insert(0, "experiment", e)
        frames.append(d)
    trials = pd.concat(frames, ignore_index=True)
    # Participant ids are unique across the run, but the two people who took
    # experiments 1 and 2 kept one id; as in the paper, the unit is a person
    # in one experiment.
    trials["participant_uid"] = (
        "e" + trials.experiment.astype(str) + "_p" + trials.participant_id.astype(str)
    )
    return trials


def collection_minutes(experiment: int) -> dict:
    """Minutes from the first Prolific poll to the first poll at the target."""
    polls = []
    for line in (RUN_ROOT / f"experiment{experiment}/data/observability.log").read_text().splitlines():
        m = _POLL.match(line)
        if m:
            polls.append((datetime.fromisoformat(m[1].replace("Z", "+00:00")), int(m[2]), int(m[3])))
    if not polls:
        raise ValueError(f"experiment {experiment}: no Prolific polls in observability.log")
    start = polls[0][0]
    full = [t for t, done, target in polls if done >= target]
    if not full:
        raise ValueError(f"experiment {experiment}: the study never reached its target")
    return {
        "experiment": experiment,
        "target": polls[0][2],
        "completed": max(done for _, done, _ in polls),
        "minutes": round((full[0] - start).total_seconds() / 60, 1),
    }


def bias_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """Per trial, whether the choice followed each rule (NaN on a tie)."""
    a, b = trials.sequence_a, trials.sequence_b
    chose_a = trials.chose_left == 1  # sequence_a is shown on the left

    def follow(fa: pd.Series, fb: pd.Series) -> pd.Series:
        return ((fa > fb) == chose_a).astype(float).where(fa != fb)

    return pd.DataFrame(
        {
            "experiment": trials.experiment,
            "participant_uid": trials.participant_uid,
            "side": chose_a.astype(float),
            "n_heads": follow(a.str.count("H"), b.str.count("H")),
            "length": follow(a.str.len(), b.str.len()),
            "alternation": follow(a.map(alternation_rate), b.map(alternation_rate)),
        }
    )


def t_summary(props: pd.Series) -> dict:
    props = props.dropna()
    n = len(props)
    if n < 2:
        return {"n_participants": n, "estimate": None, "ci_lo": None, "ci_hi": None}
    mean, se = props.mean(), props.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    return {"n_participants": n, "estimate": mean, "ci_lo": mean - half, "ci_hi": mean + half}


def main(args: Args) -> None:
    from plotnine import (
        aes,
        facet_wrap,
        geom_hline,
        geom_pointrange,
        ggplot,
        labs,
        scale_y_continuous,
        theme,
        theme_bw,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials = load_trials()

    # Stimulus reuse (H/T-consolidated unordered pairs, as in 01_response_analysis):
    # the paper's figure is distinct consolidated pairs over all pairs shown.
    shown = trials.assign(
        unordered=lambda d: [" | ".join(sorted(p)) for p in zip(d.sequence_a, d.sequence_b)],
        pair=lambda d: [canonical_pair(x, y) for x, y in zip(d.sequence_a, d.sequence_b)],
    ).drop_duplicates(["experiment", "unordered"])
    n_cells = shown.groupby("pair").experiment.nunique()
    reused = n_cells[n_cells > 1]
    per_exp = shown.groupby("experiment").agg(shown=("unordered", "size"), distinct=("pair", "nunique"))

    # Sequence lengths over each experiment's distinct sequences.
    seqs = pd.concat(
        [trials[["experiment", "sequence_a"]].rename(columns={"sequence_a": "s"}),
         trials[["experiment", "sequence_b"]].rename(columns={"sequence_b": "s"})]
    ).drop_duplicates()
    lengths = seqs.assign(length=seqs.s.str.len()).groupby("experiment").length.agg(
        ["count", "min", "median", "max"]
    )

    # Choice biases: per-participant proportion following each rule.
    bt = bias_trials(trials)
    per_person = bt.groupby(["experiment", "participant_uid"])[list(BIASES)].mean().reset_index()
    topline = {b: t_summary(per_person[b]) for b in BIASES}
    by_exp = pd.DataFrame(
        [
            {"experiment": e, "bias": b, **t_summary(g[b])}
            for e, g in per_person.groupby("experiment")
            for b in BIASES
        ]
    )
    by_exp.to_csv(args.out_dir / "biases_by_experiment.csv", index=False)

    out = {
        "n_trials": len(trials),
        "n_participant_experiments": trials.participant_uid.nunique(),
        "n_distinct_participant_ids": int(trials.participant_id.nunique()),
        "trials_per_participant": sorted(trials.groupby("participant_uid").size().unique().tolist()),
        "collection": [collection_minutes(e) for e in EXPERIMENTS],
        "stimulus_pairs": {
            "pairs_shown": len(shown),
            "distinct_pairs_ht_consolidated": int(len(n_cells)),
            "fraction_distinct": len(n_cells) / len(shown),
            "per_experiment_shown_and_distinct": {
                int(e): {"shown": int(r.shown), "distinct_ht_consolidated": int(r.distinct)}
                for e, r in per_exp.iterrows()
            },
            "pairs_in_more_than_one_experiment": sorted(reused.index.tolist()),
            "different_length_pairs": int((trials.sequence_a.str.len() != trials.sequence_b.str.len()).sum()),
        },
        "sequence_lengths_by_experiment": {
            int(e): {k: float(v) for k, v in row.items()} for e, row in lengths.iterrows()
        },
        "biases_topline": topline,
    }
    (args.out_dir / "descriptives.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

    plot_df = by_exp.dropna(subset=["estimate"]).assign(
        experiment=lambda d: d.experiment.astype(str),
        bias=lambda d: pd.Categorical(d.bias, categories=[b for b in BIASES if b in set(d.bias)]),
    )
    fig = (
        ggplot(plot_df, aes(x="experiment", y="estimate"))
        + geom_hline(yintercept=0.5, linetype="dashed", color="grey")
        + geom_pointrange(aes(ymin="ci_lo", ymax="ci_hi"))
        + facet_wrap("bias", nrow=1)
        + scale_y_continuous(limits=(0, 1))
        + labs(x="experiment", y="proportion of choices following the rule (95% CI)")
        + theme_bw()
        + theme(figure_size=(8, 3))
    )
    fig.save(args.out_dir / "biases_by_experiment.pdf", verbose=False)

    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main(tyro.cli(Args))
