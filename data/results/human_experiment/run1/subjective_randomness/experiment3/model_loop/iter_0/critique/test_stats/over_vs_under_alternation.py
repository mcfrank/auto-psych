# name: over_vs_under_alternation
# description: Proportion of times sequence A is chosen when it over-alternates (p_alts > 0.7) and B under-alternates (p_alts < 0.3).
import pandas as pd

def test_statistic(df):
    subset = df[(df['p_alts_a'] > 0.7) & (df['p_alts_b'] < 0.3)]
    if len(subset) == 0: return 0.0
    return float(subset['chose_left'].mean())
