import pandas as pd
import numpy as np

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')

def agg_local_mse2(seq):
    mses = []
    for w in [2, 3, 4]:
        if len(seq) < w: continue
        expected = w / 2.0
        # squared error normalized by max possible error (w/2)^2
        for i in range(len(seq) - w + 1):
            mses.append((seq[i:i+w].count('H') - expected)**2 / ((w/2.0)**2))
    return np.mean(mses) if mses else 0.0

df['agg_mse2_a'] = df['sequence_a'].apply(agg_local_mse2)
df['agg_mse2_b'] = df['sequence_b'].apply(agg_local_mse2)
print("Corr normalized MSE:", np.corrcoef(df['agg_mse2_a'] - df['agg_mse2_b'], df['chose_left'])[0, 1])

def local_dev(seq):
    devs = []
    # just look at window 3 absolute deviations
    for i in range(len(seq) - 2):
        devs.append(abs(seq[i:i+3].count('H') - 1.5))
    return np.mean(devs)
df['loc_dev3_a'] = df['sequence_a'].apply(local_dev)
df['loc_dev3_b'] = df['sequence_b'].apply(local_dev)
print("Loc Dev 3 Corr:", np.corrcoef(df['loc_dev3_a'] - df['loc_dev3_b'], df['chose_left'])[0, 1])

