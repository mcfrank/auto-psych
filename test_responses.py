import pandas as pd
import numpy as np

df = pd.read_csv('/Users/ben/Documents/auto-psych/data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/fewer_heads_more_random/experiment2/model_loop/responses.csv')

def eval_score(score_a, score_b):
    # If score_a > score_b, chose_left should be 1
    predicted = (score_a > score_b).astype(int)
    # Exclude ties
    mask = score_a != score_b
    acc = (predicted[mask] == df['chose_left'][mask]).mean()
    return acc

h_a = df['h_a']
h_b = df['h_b']
n_a = df['n_a']
n_b = df['n_b']

print("-h:", eval_score(-h_a, -h_b))
print("tails:", eval_score(n_a - h_a, n_b - h_b))
print("net_tails:", eval_score((n_a - h_a) - h_a, (n_b - h_b) - h_b))
print("-h^2:", eval_score(-(h_a**2), -(h_b**2)))
print("log(-h):", eval_score(-np.log(h_a+1), -np.log(h_b+1)))
