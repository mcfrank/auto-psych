import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')
print("When periodicity_a > periodicity_b, chose_left mean:", df[df['periodicity_a'] > df['periodicity_b']]['chose_left'].mean())
print("When periodicity_a < periodicity_b, chose_left mean:", df[df['periodicity_a'] < df['periodicity_b']]['chose_left'].mean())
# Also check holding p_alts diff constant
mask = (df['periodicity_a'] > df['periodicity_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.15)
print("When periodicity_a > periodicity_b and p_alts roughly equal, chose_left mean:", df[mask]['chose_left'].mean())
mask2 = (df['periodicity_a'] < df['periodicity_b']) & (abs(df['p_alts_a'] - df['p_alts_b']) <= 0.15)
print("When periodicity_a < periodicity_b and p_alts roughly equal, chose_left mean:", df[mask2]['chose_left'].mean())
