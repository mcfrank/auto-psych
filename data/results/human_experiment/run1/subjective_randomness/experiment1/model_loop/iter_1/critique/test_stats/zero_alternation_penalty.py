# name: zero_alternation_penalty
# description: Choice rate of sequence A when it has zero alternations (all identical outcomes) and B has at least one.
def test_statistic(df):
    mask = (df['p_alts_a'] == 0.0) & (df['p_alts_b'] > 0.0)
    return float(df.loc[mask, 'chose_left'].mean()) if mask.sum() > 0 else 0.5
