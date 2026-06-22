# name: extreme_under_alternation
# description: Proportion of times sequence A is chosen when it has extreme under-alternation (p_alts < 0.2) while B has moderate under-alternation (0.3 < p_alts < 0.4).
import pandas as pd

def test_statistic(df):
    subset = df[(df['p_alts_a'] < 0.2) & (df['p_alts_b'] > 0.3) & (df['p_alts_b'] < 0.4)]
    if len(subset) == 0: return 0.0
    return float(subset['chose_left'].mean())
