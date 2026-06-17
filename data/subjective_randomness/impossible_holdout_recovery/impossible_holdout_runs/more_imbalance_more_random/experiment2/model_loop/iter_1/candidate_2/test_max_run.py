import pandas as pd
import numpy as np
from scipy.optimize import minimize

df = pd.read_csv('/Users/ben/Documents/auto-psych/data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/more_imbalance_more_random/experiment2/model_loop/responses.csv')

def nll_max_run(params):
    tau = params[0]
    score_a = df['max_run_norm_a']
    score_b = df['max_run_norm_b']
    p_left = 1 / (1 + np.exp(-tau * (score_a - score_b)))
    p_left = np.clip(p_left, 1e-6, 1 - 1e-6)
    ll = df['chose_left'] * np.log(p_left) + (1 - df['chose_left']) * np.log(1 - p_left)
    return -np.sum(ll)

res_mr = minimize(nll_max_run, [1.0])
print(f"Max Run NLL: {res_mr.fun:.2f}, params: {res_mr.x}")
