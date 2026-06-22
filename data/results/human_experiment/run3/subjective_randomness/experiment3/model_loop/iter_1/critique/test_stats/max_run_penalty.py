# name: max_run_penalty
# description: The mean choice rate for sequence A when A has a larger normalized maximum run length than sequence B.
import numpy as np
import pandas as pd

def test_statistic(df):
    subset = df[df['max_run_norm_a'] > df['max_run_norm_b']]
    if len(subset) == 0: return np.nan
    return float(subset['chose_left'].mean())
