import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/responses.csv')

def s1(df): return ((df['max_run_norm_a'] > df['max_run_norm_b'] + 0.1) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2) & (abs(df['p_a'] - df['p_b']) < 0.2))
def s2(df): return ((df['periodicity_a'] > df['periodicity_b'] + 0.2) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2) & (abs(df['p_a'] - df['p_b']) < 0.2))
def s3(df): return ((df['rep_motifs_a'] > df['rep_motifs_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2) & (abs(df['p_a'] - df['p_b']) < 0.2))
def s4(df): return ((df['alt_motifs_a'] > df['alt_motifs_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2) & (abs(df['p_a'] - df['p_b']) < 0.2))
def s5(df): return ((df['n_a'] < df['n_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2) & (abs(df['p_a'] - df['p_b']) < 0.2))
def s6(df): return ((df['imbalance_a'] == 0.0) & (df['imbalance_b'] > 0.1) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2))
def s7(df): return ((df['p_alts_a'] == 1.0) & (df['p_alts_b'] < 1.0))
def s8(df): return ((df['p_alts_a'] == 0.0) & (df['p_alts_b'] > 0.0))

for i, func in enumerate([s1, s2, s3, s4, s5, s6, s7, s8], 1):
    mask = func(df)
    print(f"Stat {i}: n={mask.sum():3d}, mean={df.loc[mask, 'chose_left'].mean():.3f}")
