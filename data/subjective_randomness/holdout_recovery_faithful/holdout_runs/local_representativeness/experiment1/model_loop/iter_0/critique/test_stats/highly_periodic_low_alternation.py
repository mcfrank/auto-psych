# name: highly_periodic_low_alternation
# description: The choice rate for sequence A when it is highly periodic but has low alternation rate, testing if the model correctly penalizes periodicity when runs are few.
import numpy as np
import pandas as pd

def test_statistic(df):
    mask = (df['periodicity_a'] > 0.5) & (df['p_alts_a'] < 0.5) & (df['p_alts_b'] >= 0.5)
    if mask.sum() == 0: return 0.0
    return float(df.loc[mask, 'chose_left'].mean())
