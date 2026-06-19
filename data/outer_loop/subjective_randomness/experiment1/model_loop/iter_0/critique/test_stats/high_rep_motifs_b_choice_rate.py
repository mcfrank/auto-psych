# name: high_rep_motifs_b_choice_rate
# description: Mean chose_left when rep_motifs_b >= 3 (B contains many repeated sub-sequences); tests whether humans use repetitive-motif count as a non-randomness cue that the model omits.
def test_statistic(df):
    mask = df['rep_motifs_b'] >= 3
    sub = df[mask]
    if len(sub) == 0:
        return float('nan')
    return float(sub['chose_left'].mean())
