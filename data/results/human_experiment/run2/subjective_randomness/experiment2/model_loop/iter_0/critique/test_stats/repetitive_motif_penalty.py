# name: repetitive_motif_penalty
# description: Choice rate for the sequence with fewer repetitive motifs, conditioned on similar maximum run length and alternation rate.
def test_statistic(df):
    subset = df[(abs(df['p_alts_a'] - df['p_alts_b']) < 0.2) & 
                (abs(df['max_run_a'] - df['max_run_b']) <= 1)]
    valid = subset[subset['rep_motifs_a'] != subset['rep_motifs_b']]
    if len(valid) == 0: return 0.0
    
    chose_fewer_motifs = ((valid['rep_motifs_a'] < valid['rep_motifs_b']) & (valid['chose_left'] == 1)) | \
                         ((valid['rep_motifs_a'] > valid['rep_motifs_b']) & (valid['chose_left'] == 0))
    return float(chose_fewer_motifs.mean())
