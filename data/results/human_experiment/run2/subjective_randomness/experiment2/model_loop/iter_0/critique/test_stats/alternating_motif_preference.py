# name: alternating_motif_preference
# description: The average choice rate for the sequence with more alternating motifs, conditioned on similar overall alternation rates.
def test_statistic(df):
    subset = df[abs(df['p_alts_a'] - df['p_alts_b']) < 0.1]
    valid = subset[subset['alt_motifs_a'] != subset['alt_motifs_b']]
    if len(valid) == 0: return 0.0
    
    chose_more_alt_motifs = ((valid['alt_motifs_a'] > valid['alt_motifs_b']) & (valid['chose_left'] == 1)) | \
                            ((valid['alt_motifs_a'] < valid['alt_motifs_b']) & (valid['chose_left'] == 0))
    return float(chose_more_alt_motifs.mean())
