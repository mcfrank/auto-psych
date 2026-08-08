# name: alt_motifs_preference
# description: The mean probability of choosing the left sequence when it has strictly more alternating motifs than the right sequence.
def test_statistic(df):
    mask = df['alt_motifs_a'] > df['alt_motifs_b']
    if not mask.any(): return 0.0
    return float(df.loc[mask, 'chose_left'].mean())
