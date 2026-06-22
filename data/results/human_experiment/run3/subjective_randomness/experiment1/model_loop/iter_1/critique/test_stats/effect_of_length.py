# name: length_preference
# description: The overall probability of choosing the longer sequence when sequence lengths differ.
import numpy as np
import pandas as pd

def test_statistic(df):
    mask = df['n_a'] != df['n_b']
    if not mask.any(): return np.nan
    subset = df[mask]
    chose_longer = ((subset['n_a'] > subset['n_b']) & (subset['chose_left'] == 1)) | \
                   ((subset['n_b'] > subset['n_a']) & (subset['chose_left'] == 0))
    return float(chose_longer.mean())
