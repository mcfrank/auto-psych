# name: repeated_motifs_penalty
# description: Proportion of times sequence A is chosen when it has a higher density of repeated motifs than B, despite having similar overall alternation rates.
import pandas as pd

def test_statistic(df):
    subset = df[((df['rep_motifs_a'] / df['n_a']) > (df['rep_motifs_b'] / df['n_b']) + 0.1) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.2)]
    if len(subset) == 0: return 0.0
    return float(subset['chose_left'].mean())
