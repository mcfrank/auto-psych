# name: periodicity_effect
# description: The mean probability of choosing the left sequence when it is more periodic, excluding perfectly uniform/alternating sequences.
def test_statistic(df):
    mask = (df['periodicity_a'] > df['periodicity_b']) & (df['p_alts_a'] > 0.0) & (df['p_alts_a'] < 1.0) & (df['p_alts_b'] > 0.0) & (df['p_alts_b'] < 1.0)
    if not mask.any(): return 0.0
    return float(df.loc[mask, 'chose_left'].mean())
