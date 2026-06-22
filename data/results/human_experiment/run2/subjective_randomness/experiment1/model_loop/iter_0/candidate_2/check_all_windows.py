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
df['all_dev_b'] = df['sequence_b'].apply(all_windows_dev)
print("Corr all windows:", np.corrcoef(df['all_dev_a'] - df['all_dev_b'], df['chose_left'])[0, 1])

def weighted_all_windows(seq):
    score = 0
    total_w = 0
    # Maybe weight smaller windows more?
    for w in range(2, len(seq) + 1):
        weight = 1.0 / w
        w_mses = []
        for i in range(len(seq) - w + 1):
            prop_h = seq[i:i+w].count('H') / w
            w_mses.append((prop_h - 0.5)**2)
        score += weight * np.mean(w_mses)
        total_w += weight
    return score / total_w

df['w_all_dev_a'] = df['sequence_a'].apply(weighted_all_windows)
df['w_all_dev_b'] = df['sequence_b'].apply(weighted_all_windows)
print("Corr weighted all windows:", np.corrcoef(df['w_all_dev_a'] - df['w_all_dev_b'], df['chose_left'])[0, 1])

