# name: rep_motifs_preference
# description: The probability of choosing the sequence with more repeated motifs when p_alts difference is <= 0.15.
import numpy as np
import pandas as pd

def test_statistic(df):
    mask = (np.abs(df['p_alts_a'] - df['p_alts_b']) <= 0.15) & (df['rep_motifs_a'] != df['rep_motifs_b'])
    if not mask.any(): return np.nan
    subset = df[mask]
    chose_higher_rep = ((subset['rep_motifs_a'] > subset['rep_motifs_b']) & (subset['chose_left'] == 1)) | \
                       ((subset['rep_motifs_b'] > subset['rep_motifs_a']) & (subset['chose_left'] == 0))
    return float(chose_higher_rep.mean())
