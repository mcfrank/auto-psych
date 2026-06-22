import pandas as pd
import numpy as np

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')
for col in ['h', 'alts', 'max_run', 'rep_motifs', 'alt_motifs', 'p', 'p_alts', 'max_run_norm', 'imbalance', 'periodicity']:
    diff = df[f'{col}_a'] - df[f'{col}_b']
    corr = np.corrcoef(diff, df['chose_left'])[0, 1]
    print(f'{col}_diff: {corr:.3f}')
