# name: repeating_motifs_controlled_for_alternation
# description: Correlation between difference in repeating motifs and choice on trials with similar alternation rates, testing if motifs matter beyond runs.
import numpy as np
import pandas as pd

def test_statistic(df):
    mask = abs(df['p_alts_a'] - df['p_alts_b']) <= 0.2
    if mask.sum() < 5: return 0.0
    sub = df[mask]
    diff = sub['rep_motifs_a'] - sub['rep_motifs_b']
    if diff.std() == 0 or sub['chose_left'].std() == 0: return 0.0
    return float(np.corrcoef(diff, sub['chose_left'])[0, 1])
