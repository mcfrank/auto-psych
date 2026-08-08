# name: moderate_vs_high_alternation
# description: The mean probability of choosing the left sequence when it has moderate alternation (0.4-0.6) and the right has high (>0.8).
def test_statistic(df):
    mask = (df['p_alts_a'] >= 0.4) & (df['p_alts_a'] <= 0.6) & (df['p_alts_b'] > 0.8)
    if not mask.any(): return 0.0
    return float(df.loc[mask, 'chose_left'].mean())
