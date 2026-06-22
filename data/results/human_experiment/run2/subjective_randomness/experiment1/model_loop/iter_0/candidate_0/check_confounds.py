import pandas as pd
import numpy as np

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')
p_alts_dist_a = np.abs(df['p_alts_a'] - 0.6)
p_alts_dist_b = np.abs(df['p_alts_b'] - 0.6)

print("Corr imbalance_a vs abs(p_alts_a - 0.6):", df['imbalance_a'].corr(p_alts_dist_a))

sub = df[df['imbalance_a'] > df['imbalance_b']]
print("When A is more imbalanced, mean dist_A:", p_alts_dist_a[sub.index].mean())
print("When A is more imbalanced, mean dist_B:", p_alts_dist_b[sub.index].mean())

