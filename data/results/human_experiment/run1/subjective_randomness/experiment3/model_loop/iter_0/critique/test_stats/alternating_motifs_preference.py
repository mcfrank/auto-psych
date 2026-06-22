# name: alternating_motifs_preference
# description: Proportion of times sequence A is chosen when it has a higher density of alternating motifs than B, despite having similar overall alternation rates.
import pandas as pd

def test_statistic(df):
    subset = df[((df['alt_motifs_a'] / df['n_a']) > (df['alt_motifs_b'] / df['n_b']) + 0.1) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.2)]
    if len(subset) == 0: return 0.0
    return float(subset['chose_left'].mean())
