# name: preference_for_short_runs
# description: Choice rate for A when A has mostly very short runs (<=1.5) and B has longer runs (>=2.5), testing if multiplying by number of runs correctly captures humans' preference.
import numpy as np
import pandas as pd

def test_statistic(df):
    run_len_a = df['n_a'] / (df['alts_a'] + 1)
    run_len_b = df['n_b'] / (df['alts_b'] + 1)
    mask = (run_len_a <= 1.5) & (run_len_b >= 2.5)
    if mask.sum() == 0: return 0.0
    return float(df.loc[mask, 'chose_left'].mean())
