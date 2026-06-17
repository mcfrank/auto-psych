import pandas as pd
import numpy as np
from scipy.optimize import minimize

df = pd.read_csv('/Users/ben/Documents/auto-psych/data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/more_imbalance_more_random/experiment2/model_loop/responses.csv')

def nll_imb_over_n(params):
    tau = params[0]
    score_a = df['imbalance_a'] / df['n_a']
    score_b = df['imbalance_b'] / df['n_b']
    p_left = 1 / (1 + np.exp(-tau * (score_a - score_b)))
    p_left = np.clip(p_left, 1e-6, 1 - 1e-6)
    ll = df['chose_left'] * np.log(p_left) + (1 - df['chose_left']) * np.log(1 - p_left)
    return -np.sum(ll)

def nll_zscore(params):
    tau = params[0]
    score_a = df['imbalance_a'] * np.sqrt(df['n_a'])
    score_b = df['imbalance_b'] * np.sqrt(df['n_b'])
    p_left = 1 / (1 + np.exp(-tau * (score_a - score_b)))
    p_left = np.clip(p_left, 1e-6, 1 - 1e-6)
    ll = df['chose_left'] * np.log(p_left) + (1 - df['chose_left']) * np.log(1 - p_left)
    return -np.sum(ll)

res_ion = minimize(nll_imb_over_n, [1.0])
res_z = minimize(nll_zscore, [1.0])

print(f"Imb/N NLL: {res_ion.fun:.2f}")
print(f"Z-score NLL: {res_z.fun:.2f}")
