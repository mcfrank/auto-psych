# name: periodicity_aversion
# description: The probability of choosing the sequence with the higher periodicity score when periodicities differ.
import numpy as np
import pandas as pd

def test_statistic(df):
    mask = df['periodicity_a'] != df['periodicity_b']
    if not mask.any(): return np.nan
    subset = df[mask]
    chose_higher_periodicity = ((subset['periodicity_a'] > subset['periodicity_b']) & (subset['chose_left'] == 1)) | \
                               ((subset['periodicity_b'] > subset['periodicity_a']) & (subset['chose_left'] == 0))
    return float(chose_higher_periodicity.mean())
