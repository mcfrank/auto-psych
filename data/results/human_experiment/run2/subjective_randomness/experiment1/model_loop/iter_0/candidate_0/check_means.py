import pandas as pd

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')
print("Mean chose_left:", df['chose_left'].mean())
print("Mean chose_left when p_alts_a > p_alts_b:", df[df['p_alts_a'] > df['p_alts_b']]['chose_left'].mean())
print("Mean chose_left when p_alts_a < p_alts_b:", df[df['p_alts_a'] < df['p_alts_b']]['chose_left'].mean())
print("Mean chose_left when imbalance_a > imbalance_b:", df[df['imbalance_a'] > df['imbalance_b']]['chose_left'].mean())
print("Mean chose_left when imbalance_a < imbalance_b:", df[df['imbalance_a'] < df['imbalance_b']]['chose_left'].mean())
