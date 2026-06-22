# name: rep_motifs_preference
# description: Choice rate of sequence A when it has more repeated motifs than B, controlling for alternation rate and proportion of heads.
def test_statistic(df):
    mask = (df['rep_motifs_a'] > df['rep_motifs_b']) & \
           (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2) & \
           (abs(df['p_a'] - df['p_b']) < 0.2)
    return float(df.loc[mask, 'chose_left'].mean()) if mask.sum() > 0 else 0.5
