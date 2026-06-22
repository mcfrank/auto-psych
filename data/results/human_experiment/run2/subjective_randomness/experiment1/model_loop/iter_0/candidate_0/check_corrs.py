import pandas as pd

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')
print("Corr periodicity vs chose_left:", (df['periodicity_a'] - df['periodicity_b']).corr(df['chose_left']))
print("Corr p_alts vs chose_left:", (df['p_alts_a'] - df['p_alts_b']).corr(df['chose_left']))
print("Corr max_run vs chose_left:", (df['max_run_norm_a'] - df['max_run_norm_b']).corr(df['chose_left']))
print("Corr imbalance vs chose_left:", (df['imbalance_a'] - df['imbalance_b']).corr(df['chose_left']))

