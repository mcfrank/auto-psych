# name: high_alternation_penalty
# description: The mean choice rate for sequence A when A is highly alternating (>0.8) and B is normally alternating (0.4 to 0.6).
import numpy as np
import pandas as pd

def test_statistic(df):
    subset = df[(df['p_alts_a'] > 0.8) & (df['p_alts_b'] >= 0.4) & (df['p_alts_b'] <= 0.6)]
    if len(subset) == 0: return np.nan
    return float(subset['chose_left'].mean())
