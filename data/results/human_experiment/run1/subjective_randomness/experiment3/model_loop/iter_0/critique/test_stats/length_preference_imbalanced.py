# name: length_preference_imbalanced
# description: Proportion of trials where the longer sequence is chosen, among trials where both sequences are highly imbalanced (abs(p - 0.5) > 0.3).
import pandas as pd

def test_statistic(df):
    imbalanced_a = (abs(df['p_a'] - 0.5) > 0.3)
    imbalanced_b = (abs(df['p_b'] - 0.5) > 0.3)
    subset = df[imbalanced_a & imbalanced_b & (df['n_a'] != df['n_b'])]
    if len(subset) == 0: return 0.0
    chose_longer = ((df['n_a'] > df['n_b']) & df['chose_left']) | ((df['n_b'] > df['n_a']) & (df['chose_left'] == 0))
    return float(chose_longer.mean())
