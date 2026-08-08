# name: rep_motifs_aversion
# description: The mean probability of choosing the left sequence when it has strictly more repeating motifs than the right sequence.
def test_statistic(df):
    mask = df['rep_motifs_a'] > df['rep_motifs_b']
    if not mask.any(): return 0.0
    return float(df.loc[mask, 'chose_left'].mean())
