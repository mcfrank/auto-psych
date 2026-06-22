import pandas as pd
import numpy as np

df = pd.read_csv(
    "/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/responses.csv"
)


def stat_1(df):
    mask = (df.n_a == df.n_b) & (df.max_run_norm_a != df.max_run_norm_b)
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "max_run_norm_b"] - df.loc[mask, "max_run_norm_a"],
            df.loc[mask, "chose_left"],
        )[0, 1]
    )


def stat_2(df):
    mask = (df.n_a == df.n_b) & (df.periodicity_a != df.periodicity_b)
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "periodicity_b"] - df.loc[mask, "periodicity_a"],
            df.loc[mask, "chose_left"],
        )[0, 1]
    )


def stat_3(df):
    mask = (df.n_a == df.n_b) & (df.alt_motifs_a != df.alt_motifs_b)
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "alt_motifs_a"] - df.loc[mask, "alt_motifs_b"],
            df.loc[mask, "chose_left"],
        )[0, 1]
    )


def stat_4(df):
    mask = (df.n_a == df.n_b) & (df.rep_motifs_a != df.rep_motifs_b)
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "rep_motifs_b"] - df.loc[mask, "rep_motifs_a"],
            df.loc[mask, "chose_left"],
        )[0, 1]
    )


def stat_5(df):
    mask = df.n_a != df.n_b
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "n_a"] - df.loc[mask, "n_b"], df.loc[mask, "chose_left"]
        )[0, 1]
    )


def stat_6(df):
    # length preference when sequences are imbalanced
    mask = (df.n_a != df.n_b) & (df.imbalance_a > 0.25) & (df.imbalance_b > 0.25)
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "n_a"] - df.loc[mask, "n_b"], df.loc[mask, "chose_left"]
        )[0, 1]
    )


def stat_7(df):
    # length preference when sequences are extremely alternating
    mask = (df.n_a != df.n_b) & (df.p_alts_a > 0.6) & (df.p_alts_b > 0.6)
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "n_a"] - df.loc[mask, "n_b"], df.loc[mask, "chose_left"]
        )[0, 1]
    )


def stat_8(df):
    # slope of alts penalty in the over-alternating regime
    mask = (df.p_alts_a > 0.6) & (df.p_alts_b > 0.6) & (df.p_alts_a != df.p_alts_b)
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "p_alts_b"] - df.loc[mask, "p_alts_a"],
            df.loc[mask, "chose_left"],
        )[0, 1]
    )


for i in range(1, 9):
    print(f"stat_{i}:", eval(f"stat_{i}(df)"))
