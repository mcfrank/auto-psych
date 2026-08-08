# name: length_effect_on_randomness
# description: The overall choice rate for sequence A when A is short (length <= 4) and B is long (length == 8), testing if the model miscalibrates length.
import numpy as np
import pandas as pd

def test_statistic(df):
    mask = (df['n_a'] <= 4) & (df['n_b'] == 8)
    if mask.sum() == 0: return 0.0
    return float(df.loc[mask, 'chose_left'].mean())
