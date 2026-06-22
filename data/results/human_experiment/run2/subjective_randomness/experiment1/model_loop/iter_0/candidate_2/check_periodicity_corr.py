import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')
print("Corr periodicity_a vs p_alts_a:", df['periodicity_a'].corr(df['p_alts_a']))
print("Corr periodicity_a vs abs(p_alts_a - 0.6):", df['periodicity_a'].corr(abs(df['p_alts_a'] - 0.6)))
