import pandas as pd
import numpy as np

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/responses.csv')

def stat_max_run(df):
    mask = (df['max_run_norm_a'] - df['max_run_norm_b']) > 0.2
    return mask.sum(), df.loc[mask, 'chose_left'].mean()

def stat_periodicity(df):
    mask = (df['periodicity_a'] - df['periodicity_b']) > 0.2
    return mask.sum(), df.loc[mask, 'chose_left'].mean()

def stat_extreme_alt(df):
    mask = (df['p_alts_a'] == 1.0) & (df['p_alts_b'] < 1.0)
    return mask.sum(), df.loc[mask, 'chose_left'].mean()

def stat_alt_asymmetry(df):
    mask = (df['p_alts_a'] < 0.35) & (df['p_alts_b'] > 0.65)
    return mask.sum(), df.loc[mask, 'chose_left'].mean()

def stat_length(df):
    mask = df['n_a'] < df['n_b']
    return mask.sum(), df.loc[mask, 'chose_left'].mean()

def stat_imbalance(df):
    mask = (df['imbalance_a'] - df['imbalance_b']) > 0.2
    return mask.sum(), df.loc[mask, 'chose_left'].mean()

def stat_alt_motifs(df):
    mask = (df['alt_motifs_a'] > df['alt_motifs_b'])
    return mask.sum(), df.loc[mask, 'chose_left'].mean()

def stat_rep_motifs(df):
    mask = (df['rep_motifs_a'] > df['rep_motifs_b'])
    return mask.sum(), df.loc[mask, 'chose_left'].mean()

for name, func in [
    ('max_run', stat_max_run),
    ('periodicity', stat_periodicity),
    ('extreme_alt', stat_extreme_alt),
    ('alt_asymmetry', stat_alt_asymmetry),
    ('length', stat_length),
    ('imbalance', stat_imbalance),
    ('alt_motifs', stat_alt_motifs),
    ('rep_motifs', stat_rep_motifs)
]:
    n, mean = func(df)
    print(f"{name:15s} n={n:4d} mean={mean:.3f}")
