# name: extreme_alternation_vs_clustering
# description: Choice rate for A when A is highly alternating and B is highly clustered, testing if the multiplicative runs mechanism over-penalizes extreme alternation.
import numpy as np
import pandas as pd

def test_statistic(df):
    mask = (df['p_alts_a'] > 0.7) & (df['p_alts_b'] < 0.3)
    if mask.sum() == 0: return 0.0
    return float(df.loc[mask, 'chose_left'].mean())
