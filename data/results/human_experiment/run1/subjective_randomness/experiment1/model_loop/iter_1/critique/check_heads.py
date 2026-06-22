import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/responses.csv')
mask = (df['p_a'] > 0.5) & (df['p_b'] < 0.5) & (abs(df['imbalance_a'] - df['imbalance_b']) < 0.1)
print(f"More heads: n={mask.sum()}, mean={df.loc[mask, 'chose_left'].mean():.3f}")

mask2 = (df['p_a'] > df['p_b']) & (abs(df['imbalance_a'] - df['imbalance_b']) < 0.1) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2)
print(f"More heads (controlled): n={mask2.sum()}, mean={df.loc[mask2, 'chose_left'].mean():.3f}")
