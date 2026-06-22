import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/responses.csv')
mask = (df['periodicity_a'] == 1.0) & (df['periodicity_b'] < 1.0)
print(f"periodicity_a == 1.0: n={mask.sum()} mean={df.loc[mask, 'chose_left'].mean():.3f}")

mask2 = (df['alt_motifs_a'] == 0) & (df['alt_motifs_b'] > 0)
print(f"alt_motifs_a == 0: n={mask2.sum()} mean={df.loc[mask2, 'chose_left'].mean():.3f}")

mask3 = (df['p_a'] == 0.5) & (df['p_b'] != 0.5)
print(f"p_a == 0.5 vs not: n={mask3.sum()} mean={df.loc[mask3, 'chose_left'].mean():.3f}")
