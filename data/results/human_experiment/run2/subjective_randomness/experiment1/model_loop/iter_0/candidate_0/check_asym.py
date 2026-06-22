import pandas as pd

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')

# Let's say theta is 0.6.
# If A has p_alts=1.0 (dist=0.4), B has p_alts=0.2 (dist=0.4).
# If people penalize under-alternation more, they will choose A.
# Let's see:
df['dist_a'] = (df['p_alts_a'] - 0.6).abs()
df['dist_b'] = (df['p_alts_b'] - 0.6).abs()

# Find trials where dist_a and dist_b are similar, but one is above and one is below.
mask = (df['dist_a'] - df['dist_b']).abs() < 0.1
sub = df[mask]
# where A is above 0.6 and B is below 0.6
above_below = sub[(sub['p_alts_a'] > 0.6) & (sub['p_alts_b'] < 0.6)]
print("Count A>0.6, B<0.6 with similar distance:", len(above_below))
print("Mean chose_left:", above_below['chose_left'].mean())

