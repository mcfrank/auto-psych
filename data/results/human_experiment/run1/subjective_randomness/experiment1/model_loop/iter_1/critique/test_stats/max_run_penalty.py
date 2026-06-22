# name: max_run_penalty
# description: Choice rate of sequence A when it has a noticeably longer maximum run than B, controlling for alternation rate and proportion of heads.
def test_statistic(df):
    mask = (df['max_run_norm_a'] > df['max_run_norm_b'] + 0.1) & \
           (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2) & \
           (abs(df['p_a'] - df['p_b']) < 0.2)
    return float(df.loc[mask, 'chose_left'].mean()) if mask.sum() > 0 else 0.5
