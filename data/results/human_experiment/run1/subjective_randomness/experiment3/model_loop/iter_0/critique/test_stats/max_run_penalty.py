# name: max_run_penalty
# description: Proportion of times sequence A is chosen when it has a longer max run than B, but similar alternation rates (abs(p_alts_a - p_alts_b) <= 0.1).
import pandas as pd

def test_statistic(df):
    subset = df[(df['max_run_a'] > df['max_run_b'] + 1) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.1)]
    if len(subset) == 0: return 0.0
    return float(subset['chose_left'].mean())
