# name: extreme_imbalance_aversion
# description: The probability of choosing the perfectly imbalanced sequence (all H or all T) when paired against a non-extreme sequence.
import numpy as np
import pandas as pd

def test_statistic(df):
    mask = ((df['imbalance_a'] == 1.0) & (df['imbalance_b'] < 1.0)) | \
           ((df['imbalance_b'] == 1.0) & (df['imbalance_a'] < 1.0))
    if not mask.any(): return np.nan
    subset = df[mask]
    chose_extreme = ((subset['imbalance_a'] == 1.0) & (subset['chose_left'] == 1)) | \
                    ((subset['imbalance_b'] == 1.0) & (subset['chose_left'] == 0))
    return float(chose_extreme.mean())
