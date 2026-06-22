import pandas as pd
import numpy as np

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')

def short_windows_dev(seq):
    mses = []
    max_w = max(2, len(seq) // 2)
    for w in range(2, max_w + 1):
        for i in range(len(seq) - w + 1):
            prop_h = seq[i:i+w].count('H') / w
            mses.append((prop_h - 0.5)**2)
    return np.mean(mses)

df['short_dev_a'] = df['sequence_a'].apply(short_windows_dev)
df['short_dev_b'] = df['sequence_b'].apply(short_windows_dev)
print("Corr short windows dev:", np.corrcoef(df['short_dev_a'] - df['short_dev_b'], df['chose_left'])[0, 1])
