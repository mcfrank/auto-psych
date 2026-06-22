import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/responses.csv')

def check(name, mask):
    if mask.sum() > 20:
        print(f"{name:30s}: n={mask.sum():4d}, mean={df.loc[mask, 'chose_left'].mean():.3f}")

check('max_run_norm_a > b + 0.1', (df['max_run_norm_a'] > df['max_run_norm_b'] + 0.1) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2))
check('periodicity_a > b + 0.2', (df['periodicity_a'] > df['periodicity_b'] + 0.2) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2))
check('rep_motifs_a > b', (df['rep_motifs_a'] > df['rep_motifs_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2))
check('n_a < n_b', (df['n_a'] < df['n_b']) & (abs(df['p_a'] - df['p_b']) < 0.2) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2))

# What about heads vs tails?
check('more heads (matched imbal)', (df['p_a'] > 0.5) & (df['p_b'] < 0.5) & (abs(df['imbalance_a'] - df['imbalance_b']) < 0.1))

# What about long vs short runs with exact same alt rate?
check('same alt, diff max_run', (df['p_alts_a'] == df['p_alts_b']) & (df['max_run_norm_a'] > df['max_run_norm_b']))

# Extreme values
check('p_alts_a == 1.0', (df['p_alts_a'] == 1.0) & (df['p_alts_b'] < 1.0))
check('p_alts_a == 0.0', (df['p_alts_a'] == 0.0) & (df['p_alts_b'] > 0.0))
check('imbalance_a == 0.0', (df['imbalance_a'] == 0.0) & (df['imbalance_b'] > 0.0))
check('imbalance_a == 0.5', (df['imbalance_a'] == 0.5) & (df['imbalance_b'] < 0.5))
