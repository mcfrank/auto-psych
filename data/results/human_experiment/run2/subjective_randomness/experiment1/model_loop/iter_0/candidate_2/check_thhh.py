import pandas as pd
df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')
periodic_seqs = df[df['periodicity_a'] == 1.0]
print(periodic_seqs[['sequence_a', 'sequence_b', 'periodicity_a', 'periodicity_b', 'chose_left']].head(10))
