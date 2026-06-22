import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/responses.csv')
mask_zero = (df['p_alts_a'] == 0.0) & (df['p_alts_b'] > 0.0)
print(f"p_alts_a == 0.0: n={mask_zero.sum()} mean={df.loc[mask_zero, 'chose_left'].mean():.3f}")

mask_imbal = (df['imbalance_a'] == 0.0) & (df['imbalance_b'] > 0.0)
print(f"imbalance_a == 0.0: n={mask_imbal.sum()} mean={df.loc[mask_imbal, 'chose_left'].mean():.3f}")

mask_len = (df['n_a'] == 2) & (df['n_b'] > 2)
print(f"n_a == 2: n={mask_len.sum()} mean={df.loc[mask_len, 'chose_left'].mean():.3f}")
