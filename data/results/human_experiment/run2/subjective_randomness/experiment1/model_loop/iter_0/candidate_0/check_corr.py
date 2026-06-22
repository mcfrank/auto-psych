import pandas as pd
import numpy as np

df = pd.read_csv('/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/responses.csv')
p_diff = df['periodicity_a'] - df['periodicity_b']
corr = p_diff.corr(df['chose_left'])
print(f"Correlation between periodicity_a - periodicity_b and chose_left: {corr}")

