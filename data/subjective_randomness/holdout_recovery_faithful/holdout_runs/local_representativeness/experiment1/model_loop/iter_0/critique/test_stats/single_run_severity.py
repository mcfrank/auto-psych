# name: single_run_severity
# description: Choice rate for A when A consists of exactly one run while B has multiple, testing if the single run multiplier under-penalizes total uniformity.
import numpy as np
import pandas as pd

def test_statistic(df):
    mask = (df['max_run_a'] == df['n_a']) & (df['max_run_b'] < df['n_b'])
    if mask.sum() == 0: return 0.0
    return float(df.loc[mask, 'chose_left'].mean())
