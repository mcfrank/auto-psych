import pandas as pd

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')
print(df[['sequence_a', 'periodicity_a', 'p_alts_a', 'imbalance_a']].drop_duplicates().head(20))
