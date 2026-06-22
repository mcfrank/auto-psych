# name: over_alternation_penalty
# description: The correlation between choice and p_alts difference specifically for sequences that both have p_alts > 0.6.
import numpy as np
import pandas as pd

def test_statistic(df):
    mask = (df['p_alts_a'] > 0.6) & (df['p_alts_b'] > 0.6)
    if mask.sum() < 5: return np.nan
    subset = df[mask]
    corr = subset['chose_left'].corr(subset['p_alts_a'] - subset['p_alts_b'])
    return float(corr) if not pd.isna(corr) else 0.0
