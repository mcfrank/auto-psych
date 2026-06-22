import pandas as pd
import numpy as np

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')

def w4_dev(seq):
    mses = []
    max_w = min(4, len(seq))
    for w in range(2, max_w + 1):
        for i in range(len(seq) - w + 1):
            prop_h = seq[i:i+w].count('H') / w
            mses.append((prop_h - 0.5)**2)
    return np.mean(mses)

df['w4_dev_a'] = df['sequence_a'].apply(w4_dev)
df['w4_dev_b'] = df['sequence_b'].apply(w4_dev)
print("Corr up to w=4 dev:", np.corrcoef(df['w4_dev_a'] - df['w4_dev_b'], df['chose_left'])[0, 1])

