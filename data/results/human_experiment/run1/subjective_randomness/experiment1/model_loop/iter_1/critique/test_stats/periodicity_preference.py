# name: periodicity_preference
# description: Choice rate of sequence A when it is more periodic than B, controlling for alternation rate and proportion of heads.
def test_statistic(df):
    mask = (df['periodicity_a'] > df['periodicity_b'] + 0.2) & \
           (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2) & \
           (abs(df['p_a'] - df['p_b']) < 0.2)
    return float(df.loc[mask, 'chose_left'].mean()) if mask.sum() > 0 else 0.5
