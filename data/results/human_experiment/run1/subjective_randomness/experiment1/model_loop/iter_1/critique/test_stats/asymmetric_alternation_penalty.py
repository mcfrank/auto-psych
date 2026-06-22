# name: asymmetric_alternation_penalty
# description: Choice rate of sequence A when it has very low alternation (<0.35) and B has very high alternation (>0.65), to test if the model's symmetric quadratic penalty is adequate.
def test_statistic(df):
    mask = (df['p_alts_a'] < 0.35) & (df['p_alts_b'] > 0.65)
    return float(df.loc[mask, 'chose_left'].mean()) if mask.sum() > 0 else 0.5
