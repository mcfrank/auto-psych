import pandas as pd
import numpy as np

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')

def all_windows_abs_dev(seq):
    devs = []
    for w in range(2, len(seq) + 1):
        for i in range(len(seq) - w + 1):
            prop_h = seq[i:i+w].count('H') / w
            devs.append(abs(prop_h - 0.5))
    return np.mean(devs)

df['all_abs_dev_a'] = df['sequence_a'].apply(all_windows_abs_dev)
df['all_abs_dev_b'] = df['sequence_b'].apply(all_windows_abs_dev)
print("Corr all windows abs dev:", np.corrcoef(df['all_abs_dev_a'] - df['all_abs_dev_b'], df['chose_left'])[0, 1])
