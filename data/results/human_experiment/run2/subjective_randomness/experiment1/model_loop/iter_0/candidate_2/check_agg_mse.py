import pandas as pd
import numpy as np

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')

def agg_local_mse(seq):
    mses = []
    for w in [2, 3, 4]:
        if len(seq) < w: continue
        expected = w / 2.0
        for i in range(len(seq) - w + 1):
            mses.append((seq[i:i+w].count('H') - expected)**2 / (w/2.0)) # normalize by expected? or just squared error?
    return np.mean(mses) if mses else 0.0

df['agg_mse_a'] = df['sequence_a'].apply(agg_local_mse)
df['agg_mse_b'] = df['sequence_b'].apply(agg_local_mse)
print("Corr:", np.corrcoef(df['agg_mse_a'] - df['agg_mse_b'], df['chose_left'])[0, 1])

def local_representativeness(seq, w=4):
    if len(seq) < w: return 0.0
    # absolute deviation from expected 0.5 ratio
    devs = []
    for i in range(len(seq) - w + 1):
        devs.append(abs(seq[i:i+w].count('H')/w - 0.5))
    return np.mean(devs)

df['loc_rep_a'] = df['sequence_a'].apply(lambda x: local_representativeness(x, 4))
df['loc_rep_b'] = df['sequence_b'].apply(lambda x: local_representativeness(x, 4))
print("Loc Rep w=4 Corr:", np.corrcoef(df['loc_rep_a'] - df['loc_rep_b'], df['chose_left'])[0, 1])

