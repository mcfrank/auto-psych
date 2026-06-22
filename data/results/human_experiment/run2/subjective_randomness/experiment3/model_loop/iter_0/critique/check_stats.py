import pandas as pd
import numpy as np

df = pd.read_csv(
    "/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/responses.csv"
)


def stat_max_run(df):
    mask = (abs(df.p_alts_a - df.p_alts_b) < 0.1) & (
        df.max_run_norm_a != df.max_run_norm_b
    )
    if mask.sum() == 0:
        return 0
    return np.corrcoef(
        df.loc[mask, "max_run_norm_b"] - df.loc[mask, "max_run_norm_a"],
        df.loc[mask, "chose_left"],
    )[0, 1]


def stat_periodicity(df):
    mask = (abs(df.p_alts_a - df.p_alts_b) < 0.1) & (
        df.periodicity_a != df.periodicity_b
    )
    if mask.sum() == 0:
        return 0
    return np.corrcoef(
        df.loc[mask, "periodicity_b"] - df.loc[mask, "periodicity_a"],
        df.loc[mask, "chose_left"],
    )[0, 1]


def stat_length(df):
    # Does length preference interact with imbalance?
    # Model: score_a - score_b = n_a*(ev - pen_a) - n_b*(ev - pen_b)
    # If penalty is high, n_a * (ev - pen_a) could be strongly negative.
    mask = (df.imbalance_a > 0.5) & (df.imbalance_b > 0.5) & (df.n_a != df.n_b)
    if mask.sum() == 0:
        return 0
    return np.corrcoef(
        df.loc[mask, "n_a"] - df.loc[mask, "n_b"], df.loc[mask, "chose_left"]
    )[0, 1]


def stat_rep_motifs(df):
    mask = (abs(df.max_run_norm_a - df.max_run_norm_b) < 0.1) & (
        df.rep_motifs_a != df.rep_motifs_b
    )
    if mask.sum() == 0:
        return 0
    return np.corrcoef(
        df.loc[mask, "rep_motifs_b"] - df.loc[mask, "rep_motifs_a"],
        df.loc[mask, "chose_left"],
    )[0, 1]


def stat_alt_motifs(df):
    mask = df.alt_motifs_a != df.alt_motifs_b
    return np.corrcoef(
        df.loc[mask, "alt_motifs_a"] - df.loc[mask, "alt_motifs_b"],
        df.loc[mask, "chose_left"],
    )[0, 1]


print("Max run:", stat_max_run(df))
print("Periodicity:", stat_periodicity(df))
print("Length (high imb):", stat_length(df))
print("Rep motifs:", stat_rep_motifs(df))
print("Alt motifs:", stat_alt_motifs(df))
