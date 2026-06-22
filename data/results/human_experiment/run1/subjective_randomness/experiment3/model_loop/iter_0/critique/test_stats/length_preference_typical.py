# name: length_preference_typical
# description: Proportion of trials where the longer sequence is chosen, among trials where both sequences are typical (0.3 <= p <= 0.7, 0.3 <= p_alts <= 0.7).
import pandas as pd

def test_statistic(df):
    typical_a = (df['p_a'] >= 0.3) & (df['p_a'] <= 0.7) & (df['p_alts_a'] >= 0.3) & (df['p_alts_a'] <= 0.7)
    typical_b = (df['p_b'] >= 0.3) & (df['p_b'] <= 0.7) & (df['p_alts_b'] >= 0.3) & (df['p_alts_b'] <= 0.7)
    subset = df[typical_a & typical_b & (df['n_a'] != df['n_b'])]
    if len(subset) == 0: return 0.0
    chose_longer = ((df['n_a'] > df['n_b']) & df['chose_left']) | ((df['n_b'] > df['n_a']) & (df['chose_left'] == 0))
    return float(chose_longer.mean())
