import pandas as pd
import numpy as np

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')

def safe_mean(mask):
    if mask.sum() > 0:
        return df.loc[mask, 'chose_left'].mean()
    return np.nan

print("1.", safe_mean((df['max_run_a'] > df['max_run_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.15)))
print("2.", safe_mean((df['periodicity_a'] > df['periodicity_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.15)))
print("3.", safe_mean((df['rep_motifs_a'] > df['rep_motifs_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.15)))
print("4.", safe_mean((df['p_alts_a'] > 0.7) & (df['p_alts_b'] < 0.5)))
print("5.", safe_mean((df['n_a'] > df['n_b']) & (df['imbalance_a'] == df['imbalance_b']) & (df['imbalance_a'] > 0.1)))
print("6.", np.corrcoef(df['max_run_a'] - df['max_run_b'], df['chose_left'])[0, 1])
print("7.", safe_mean((df['alt_motifs_a'] > df['alt_motifs_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.15)))
print("8.", np.corrcoef(df['periodicity_a'] - df['periodicity_b'], df['chose_left'])[0, 1])
