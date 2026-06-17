import pandas as pd
import numpy as np
from scipy.optimize import minimize

df = pd.read_csv('/Users/ben/Documents/auto-psych/data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/more_imbalance_more_random/experiment2/model_loop/responses.csv')

def nll_ratio(params):
    tau = params[0]
    # safe ratio
    score_a = np.log(df['imbalance_a'] + 1e-3)
    score_b = np.log(df['imbalance_b'] + 1e-3)
    p_left = 1 / (1 + np.exp(-tau * (score_a - score_b)))
    p_left = np.clip(p_left, 1e-6, 1 - 1e-6)
    ll = df['chose_left'] * np.log(p_left) + (1 - df['chose_left']) * np.log(1 - p_left)
    return -np.sum(ll)

res_ratio = minimize(nll_ratio, [1.0])
print(f"Ratio NLL: {res_ratio.fun:.2f}, params: {res_ratio.x}")
