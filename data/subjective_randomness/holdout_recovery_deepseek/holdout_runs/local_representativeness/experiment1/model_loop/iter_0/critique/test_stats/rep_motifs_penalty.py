# name: rep_motifs_penalty
# description: Net probability of choosing the left sequence when the left has more repeated sub-pattern motifs but not more alternation, probing detection of patterned repetition.
# description: Conditioned on trials where repeated-motif counts are unequal.
def test_statistic(df):
    a = df.loc[df['rep_motifs_a'] > df['rep_motifs_b']]
    b = df.loc[df['rep_motifs_a'] < df['rep_motifs_b']]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return a['chose_left'].mean() - b['chose_left'].mean()
