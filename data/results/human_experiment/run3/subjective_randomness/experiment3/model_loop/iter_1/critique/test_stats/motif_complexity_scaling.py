# name: motif_complexity_scaling
# description: The mean choice rate for A when A has more alternating motifs but fewer repeating motifs than B.
import numpy as np
import pandas as pd

def test_statistic(df):
    subset = df[(df['alt_motifs_a'] > df['alt_motifs_b']) & (df['rep_motifs_a'] < df['rep_motifs_b'])]
    if len(subset) == 0: return np.nan
    return float(subset['chose_left'].mean())
