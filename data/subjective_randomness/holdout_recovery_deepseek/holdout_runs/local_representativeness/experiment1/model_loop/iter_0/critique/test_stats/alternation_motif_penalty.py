# name: alternation_motif_penalty
# description: Net probability of choosing the left sequence when the left has more strict-alternation motifs than the right, conditioning on unequal alternation-motif counts but similar total alternation rate.
def test_statistic(df):
    a = df.loc[df['alt_motifs_a'] > df['alt_motifs_b']]
    b = df.loc[df['alt_motifs_a'] < df['alt_motifs_b']]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return a['chose_left'].mean() - b['chose_left'].mean()
