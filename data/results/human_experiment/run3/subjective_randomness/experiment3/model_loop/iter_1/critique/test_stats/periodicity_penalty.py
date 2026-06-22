# name: periodicity_penalty
# description: The mean choice rate for sequence A when A has higher periodicity than sequence B.
import numpy as np
import pandas as pd

def test_statistic(df):
    subset = df[df['periodicity_a'] > df['periodicity_b']]
    if len(subset) == 0: return np.nan
    return float(subset['chose_left'].mean())
