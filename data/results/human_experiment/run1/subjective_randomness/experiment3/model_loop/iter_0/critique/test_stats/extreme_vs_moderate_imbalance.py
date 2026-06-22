# name: extreme_vs_moderate_imbalance
# description: Proportion of times sequence A is chosen when it has extreme imbalance (abs(p - 0.5) > 0.35) while B has moderate imbalance (0.15 < abs(p - 0.5) < 0.25).
import pandas as pd

def test_statistic(df):
    subset = df[(abs(df['p_a'] - 0.5) > 0.35) & (abs(df['p_b'] - 0.5) > 0.15) & (abs(df['p_b'] - 0.5) < 0.25)]
    if len(subset) == 0: return 0.0
    return float(subset['chose_left'].mean())
