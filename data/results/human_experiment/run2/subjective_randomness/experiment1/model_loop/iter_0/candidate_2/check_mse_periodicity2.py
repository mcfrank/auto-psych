import pandas as pd
import numpy as np

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')

def all_windows_dev(seq):
    mses = []
    for w in range(2, len(seq) + 1):
        for i in range(len(seq) - w + 1):
            prop_h = seq[i:i+w].count('H') / w
            mses.append((prop_h - 0.5)**2)
    return np.mean(mses)

df['all_dev_a'] = df['sequence_a'].apply(all_windows_dev)
print("Corr all_dev_a vs periodicity_a:", np.corrcoef(df['all_dev_a'], df['periodicity_a'])[0, 1])
