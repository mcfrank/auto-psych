import pandas as pd
import numpy as np

df = pd.read_csv(
    "/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/responses.csv"
)


def stat_imb_short(df):
    mask = (df.n_a <= 5) & (df.n_b <= 5) & (df.imbalance_a != df.imbalance_b)
    if mask.sum() == 0:
        return 0
    return np.corrcoef(
        df.loc[mask, "imbalance_b"] - df.loc[mask, "imbalance_a"],
        df.loc[mask, "chose_left"],
    )[0, 1]


def stat_imb_long(df):
    mask = (df.n_a >= 7) & (df.n_b >= 7) & (df.imbalance_a != df.imbalance_b)
    if mask.sum() == 0:
        return 0
    return np.corrcoef(
        df.loc[mask, "imbalance_b"] - df.loc[mask, "imbalance_a"],
        df.loc[mask, "chose_left"],
    )[0, 1]


print("Imb short:", stat_imb_short(df))
print("Imb long:", stat_imb_long(df))


def stat_alt_short(df):
    mask = (df.n_a <= 5) & (df.n_b <= 5) & (df.p_alts_a != df.p_alts_b)
    if mask.sum() == 0:
        return 0
    return np.corrcoef(
        df.loc[mask, "p_alts_b"] - df.loc[mask, "p_alts_a"], df.loc[mask, "chose_left"]
    )[0, 1]


def stat_alt_long(df):
    mask = (df.n_a >= 7) & (df.n_b >= 7) & (df.p_alts_a != df.p_alts_b)
    if mask.sum() == 0:
        return 0
    return np.corrcoef(
        df.loc[mask, "p_alts_b"] - df.loc[mask, "p_alts_a"], df.loc[mask, "chose_left"]
    )[0, 1]


print("Alt short:", stat_alt_short(df))
print("Alt long:", stat_alt_long(df))
