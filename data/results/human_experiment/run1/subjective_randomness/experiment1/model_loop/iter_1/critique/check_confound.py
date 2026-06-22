import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/responses.csv')
mask = (df['imbalance_a'] == 0.5) & (df['imbalance_b'] < 0.5)
print(df.loc[mask, ['p_alts_a', 'p_alts_b']].describe())
