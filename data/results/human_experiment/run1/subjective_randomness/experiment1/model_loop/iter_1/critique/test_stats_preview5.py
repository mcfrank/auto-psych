import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/responses.csv')
mask = (df['alt_motifs_a'] == 0) & (df['alt_motifs_b'] > 0) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2)
print(f"alt_motifs_a == 0 and similar p_alts: n={mask.sum()} mean={df.loc[mask, 'chose_left'].mean():.3f}")

mask = (df['n_a'] == df['n_b']) & (df['h_a'] == df['h_b']) & (df['alts_a'] == df['alts_b']) & (df['sequence_a'] != df['sequence_b'])
print(f"perfectly matched p and alt, different sequence: n={mask.sum()} mean={df.loc[mask, 'chose_left'].mean():.3f}")
if mask.sum() > 0:
    print(df.loc[mask, ['sequence_a', 'sequence_b', 'chose_left']].head(10))
