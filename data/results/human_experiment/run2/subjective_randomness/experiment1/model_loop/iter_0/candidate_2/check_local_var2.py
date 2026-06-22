import pandas as pd
import numpy as np

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')

def local_h_var(seq, w):
    if len(seq) < w: return 0.0
    counts = []
    for i in range(len(seq) - w + 1):
        counts.append(seq[i:i+w].count('H'))
    return np.var(counts)

for w in [2, 3, 4]:
    df[f'h_var_{w}_a'] = df['sequence_a'].apply(lambda x: local_h_var(x, w))
    df[f'h_var_{w}_b'] = df['sequence_b'].apply(lambda x: local_h_var(x, w))
    diff = df[f'h_var_{w}_a'] - df[f'h_var_{w}_b']
    corr = np.corrcoef(diff, df['chose_left'])[0, 1]
    print(f"w={w} Corr: {corr:.3f}")
