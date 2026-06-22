import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/responses.csv')
mask = (df['max_run_norm_a'] > df['max_run_norm_b'] + 0.1) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2)
print(f"max_run with similar p_alts: n={mask.sum()} mean={df.loc[mask, 'chose_left'].mean():.3f}")

mask2 = (df['periodicity_a'] > df['periodicity_b'] + 0.2) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2)
print(f"periodicity with similar p_alts: n={mask2.sum()} mean={df.loc[mask2, 'chose_left'].mean():.3f}")

mask3 = (df['alt_motifs_a'] > df['alt_motifs_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2)
print(f"alt_motifs with similar p_alts: n={mask3.sum()} mean={df.loc[mask3, 'chose_left'].mean():.3f}")

mask4 = (df['rep_motifs_a'] > df['rep_motifs_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2)
print(f"rep_motifs with similar p_alts: n={mask4.sum()} mean={df.loc[mask4, 'chose_left'].mean():.3f}")

mask5 = (df['n_a'] < df['n_b']) & (abs(df['p_a'] - df['p_b']) < 0.2) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2)
print(f"shorter length with similar features: n={mask5.sum()} mean={df.loc[mask5, 'chose_left'].mean():.3f}")
