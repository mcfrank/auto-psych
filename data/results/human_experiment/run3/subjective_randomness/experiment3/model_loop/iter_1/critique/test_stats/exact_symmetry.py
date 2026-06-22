# name: exact_symmetry
# description: The mean choice rate for sequence A when A is exactly symmetric (imbalance 0) and B is slightly imbalanced (>0 to 0.3).
import numpy as np
import pandas as pd

def test_statistic(df):
    subset = df[(df['imbalance_a'] == 0) & (df['imbalance_b'] > 0) & (df['imbalance_b'] <= 0.3)]
    if len(subset) == 0: return np.nan
    return float(subset['chose_left'].mean())
