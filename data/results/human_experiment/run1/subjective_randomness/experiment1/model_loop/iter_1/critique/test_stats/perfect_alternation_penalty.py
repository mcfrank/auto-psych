# name: perfect_alternation_penalty
# description: Choice rate of sequence A when it perfectly alternates (p_alts=1.0) and B does not.
def test_statistic(df):
    mask = (df['p_alts_a'] == 1.0) & (df['p_alts_b'] < 1.0)
    return float(df.loc[mask, 'chose_left'].mean()) if mask.sum() > 0 else 0.5
