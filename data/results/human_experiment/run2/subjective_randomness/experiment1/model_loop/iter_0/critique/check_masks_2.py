import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')

def check_mask(mask_name, mask):
    print(f"{mask_name}: {mask.sum()} trials, mean chose_left = {df.loc[mask, 'chose_left'].mean():.3f}")

check_mask("max_run_a > max_run_b | p_alts diff <= 0.15", (df['max_run_a'] > df['max_run_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.15))
check_mask("periodicity_a > periodicity_b | p_alts diff <= 0.15", (df['periodicity_a'] > df['periodicity_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.15))
check_mask("rep_motifs_a > rep_motifs_b | p_alts diff <= 0.15", (df['rep_motifs_a'] > df['rep_motifs_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.15))
check_mask("alt_motifs_a > alt_motifs_b | p_alts diff <= 0.15", (df['alt_motifs_a'] > df['alt_motifs_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.15))
check_mask("p_alts_a > 0.7 & p_alts_b < 0.5", (df['p_alts_a'] > 0.7) & (df['p_alts_b'] < 0.5))
check_mask("p_alts_a < 0.5 & p_alts_b > 0.7", (df['p_alts_a'] < 0.5) & (df['p_alts_b'] > 0.7))
check_mask("n_a > n_b", (df['n_a'] > df['n_b']))
check_mask("n_a < n_b", (df['n_a'] < df['n_b']))
check_mask("max_run_norm_a > max_run_norm_b", df['max_run_norm_a'] > df['max_run_norm_b'])

