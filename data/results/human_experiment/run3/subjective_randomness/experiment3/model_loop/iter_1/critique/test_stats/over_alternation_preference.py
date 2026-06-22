# name: over_alternation_preference
# description: The mean choice rate for sequence A when A is slightly over-alternating (0.6 to 0.8) and B is near fair alternation (0.3 to 0.55).
import numpy as np
import pandas as pd

def test_statistic(df):
    subset = df[(df['p_alts_a'] >= 0.6) & (df['p_alts_a'] <= 0.8) & (df['p_alts_b'] >= 0.3) & (df['p_alts_b'] <= 0.55)]
    if len(subset) == 0: return np.nan
    return float(subset['chose_left'].mean())
