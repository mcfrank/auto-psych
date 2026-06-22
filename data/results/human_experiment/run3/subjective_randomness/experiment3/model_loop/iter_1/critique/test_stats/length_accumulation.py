# name: length_accumulation
# description: The mean choice rate for sequence A when sequence A is strictly longer than sequence B.
import numpy as np
import pandas as pd

def test_statistic(df):
    subset = df[df['n_a'] > df['n_b']]
    if len(subset) == 0: return np.nan
    return float(subset['chose_left'].mean())
