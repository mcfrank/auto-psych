import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')
mask = (df['periodicity_a'] > df['periodicity_b']) & (df['max_run_a'] == df['max_run_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.15)
print("When periodicity_a > periodicity_b, max_run equal, p_alts equal, chose_left mean:", df[mask]['chose_left'].mean())
mask2 = (df['periodicity_a'] < df['periodicity_b']) & (df['max_run_a'] == df['max_run_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.15)
print("When periodicity_a < periodicity_b, max_run equal, p_alts equal, chose_left mean:", df[mask2]['chose_left'].mean())
