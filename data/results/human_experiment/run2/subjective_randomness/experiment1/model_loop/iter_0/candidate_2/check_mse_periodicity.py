import pandas as pd
import numpy as np
from check_agg_mse2 import agg_local_mse2

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')
df['agg_mse2_a'] = df['sequence_a'].apply(agg_local_mse2)
print("Corr agg_mse2_a vs periodicity_a:", np.corrcoef(df['agg_mse2_a'], df['periodicity_a'])[0, 1])

# Also check correlation with choose left for max_run
print("Corr max_run_diff:", np.corrcoef(df['max_run_a'] - df['max_run_b'], df['chose_left'])[0, 1])
