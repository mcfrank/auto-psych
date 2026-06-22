# name: extreme_imbalance
# description: The mean choice rate for sequence A when A is highly imbalanced (>=0.75) and B is moderately imbalanced (0.2 to 0.5).
import numpy as np
import pandas as pd

def test_statistic(df):
    subset = df[(df['imbalance_a'] >= 0.75) & (df['imbalance_b'] > 0.2) & (df['imbalance_b'] <= 0.5)]
    if len(subset) == 0: return np.nan
    return float(subset['chose_left'].mean())
