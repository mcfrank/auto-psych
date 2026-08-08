# name: symmetry_effect
# description: The mean probability of choosing the left sequence when it is more symmetric (sym1_a > sym1_b).
def test_statistic(df):
    mask = df['sym1_a'] > df['sym1_b']
    if not mask.any(): return 0.0
    return float(df.loc[mask, 'chose_left'].mean())
