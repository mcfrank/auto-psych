import pandas as pd

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')

def check_mask(mask_name, mask):
    print(f"{mask_name}: {mask.sum()} trials, mean chose_left = {df.loc[mask, 'chose_left'].mean():.3f}")

# 1. max run difference when p_alts is similar
check_mask("max_run_a > max_run_b, close p_alts and imbalance", (df['max_run_a'] > df['max_run_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.15) & (abs(df['imbalance_a'] - df['imbalance_b']) <= 0.15))

# 2. over-alternating vs under-alternating.
# Does the model penalize p_alts > 0.8 the same as p_alts < 0.3?
check_mask("a is over-alternating (>0.7), b is under-alternating (<0.3)", (df['p_alts_a'] > 0.7) & (df['p_alts_b'] < 0.4))

# 3. periodicity
check_mask("a has high periodicity, b has low periodicity", (df['periodicity_a'] > df['periodicity_b'] + 0.2))

# 4. sequence length preference (n_a < n_b) when proportions are equal
check_mask("n_a < n_b, exact same proportions", (df['n_a'] < df['n_b']) & (df['p_a'] == df['p_b']) & (df['p_alts_a'] == df['p_alts_b']))
check_mask("n_a != n_b", (df['n_a'] != df['n_b']))

# 5. rep_motifs (e.g. HHHH or TTTT)
check_mask("rep_motifs_a > rep_motifs_b, close p_alts", (df['rep_motifs_a'] > df['rep_motifs_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2))

# 6. alt_motifs (e.g. HTHT)
check_mask("alt_motifs_a > alt_motifs_b, close p_alts", (df['alt_motifs_a'] > df['alt_motifs_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) < 0.2))

# 7. extreme imbalance (all heads or all tails) vs moderate imbalance
check_mask("extreme imbalance A vs moderate B", (df['imbalance_a'] == 1.0) & (df['imbalance_b'] < 1.0))

