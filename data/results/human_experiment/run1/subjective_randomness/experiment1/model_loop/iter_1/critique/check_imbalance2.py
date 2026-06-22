import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/responses.csv')
mask = (df['imbalance_a'] == 1.0) & (df['imbalance_b'] < 1.0)
print(f"imbalance_a == 1.0: n={mask.sum()} mean={df.loc[mask, 'chose_left'].mean():.3f}")
