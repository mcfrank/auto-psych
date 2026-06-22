import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/responses.csv')

funcs = [
    lambda df: (df['max_run_norm_a'] > df['max_run_norm_b'] + 0.1) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2) & (abs(df['p_a'] - df['p_b']) < 0.2),
    lambda df: (df['periodicity_a'] > df['periodicity_b'] + 0.2) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2) & (abs(df['p_a'] - df['p_b']) < 0.2),
    lambda df: (df['rep_motifs_a'] > df['rep_motifs_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2) & (abs(df['p_a'] - df['p_b']) < 0.2),
    lambda df: (df['n_a'] < df['n_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2) & (abs(df['p_a'] - df['p_b']) < 0.2),
    lambda df: (df['p_alts_a'] == 0.0) & (df['p_alts_b'] > 0.0),
    lambda df: (df['p_alts_a'] == 1.0) & (df['p_alts_b'] < 1.0),
    lambda df: (df['imbalance_a'] == 0.5) & (df['imbalance_b'] < 0.5),
    lambda df: (df['p_alts_a'] < 0.35) & (df['p_alts_b'] > 0.65),
]

for i, f in enumerate(funcs, 1):
    mask = f(df)
    print(f"Stat {i}: n={mask.sum()}, mean={df.loc[mask, 'chose_left'].mean():.3f}")
