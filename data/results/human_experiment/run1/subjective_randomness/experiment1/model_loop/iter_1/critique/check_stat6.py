import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/responses.csv')
mask = (df['imbalance_a'] == 0.0) & (df['imbalance_b'] > 0.0) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2)
print(f"Perfect balance: n={mask.sum()}, mean={df.loc[mask, 'chose_left'].mean():.3f}")

mask2 = (df['imbalance_a'] == 0.0) & (df['imbalance_b'] > 0.0)
print(f"Perfect balance (uncontrolled): n={mask2.sum()}, mean={df.loc[mask2, 'chose_left'].mean():.3f}")
