# name: max_run_preference
# description: The probability of choosing the sequence with the strictly smaller max_run, among trials where p_alts difference is <= 0.15.
import numpy as np
import pandas as pd

def test_statistic(df):
    mask = (np.abs(df['p_alts_a'] - df['p_alts_b']) <= 0.15) & (df['max_run_a'] != df['max_run_b'])
    if not mask.any(): return np.nan
    subset = df[mask]
    chose_smaller_max_run = ((subset['max_run_a'] < subset['max_run_b']) & (subset['chose_left'] == 1)) | \
                            ((subset['max_run_b'] < subset['max_run_a']) & (subset['chose_left'] == 0))
    return float(chose_smaller_max_run.mean())
