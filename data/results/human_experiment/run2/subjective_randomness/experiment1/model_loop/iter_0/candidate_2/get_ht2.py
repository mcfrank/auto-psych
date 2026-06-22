import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')
df = df[['sequence_a', 'periodicity_a', 'p_alts_a']].drop_duplicates().sort_values('periodicity_a', ascending=False)
print(df.head(20))
