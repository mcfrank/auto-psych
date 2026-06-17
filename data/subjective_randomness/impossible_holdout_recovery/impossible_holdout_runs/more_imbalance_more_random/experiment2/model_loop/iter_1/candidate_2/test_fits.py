import pandas as pd
import numpy as np
from scipy.optimize import minimize

df = pd.read_csv('/Users/ben/Documents/auto-psych/data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/more_imbalance_more_random/experiment2/model_loop/responses.csv')

def nll_softmax(params):
    tau = params[0]
    p_left = 1 / (1 + np.exp(-tau * (df['imbalance_a'] - df['imbalance_b'])))
    p_left = np.clip(p_left, 1e-6, 1 - 1e-6)
    ll = df['chose_left'] * np.log(p_left) + (1 - df['chose_left']) * np.log(1 - p_left)
    return -np.sum(ll)

def nll_lapse(params):
    tau, lapse = params
    p_left = 1 / (1 + np.exp(-tau * (df['imbalance_a'] - df['imbalance_b'])))
    p_left = (1 - lapse) * p_left + lapse * 0.5
    p_left = np.clip(p_left, 1e-6, 1 - 1e-6)
    ll = df['chose_left'] * np.log(p_left) + (1 - df['chose_left']) * np.log(1 - p_left)
    return -np.sum(ll)

def nll_bias(params):
    tau, bias = params
    p_left = 1 / (1 + np.exp(-tau * (df['imbalance_a'] - df['imbalance_b']) - bias))
    p_left = np.clip(p_left, 1e-6, 1 - 1e-6)
    ll = df['chose_left'] * np.log(p_left) + (1 - df['chose_left']) * np.log(1 - p_left)
    return -np.sum(ll)

def nll_power(params):
    tau, power = params
    p_left = 1 / (1 + np.exp(-tau * (df['imbalance_a']**power - df['imbalance_b']**power)))
    p_left = np.clip(p_left, 1e-6, 1 - 1e-6)
    ll = df['chose_left'] * np.log(p_left) + (1 - df['chose_left']) * np.log(1 - p_left)
    return -np.sum(ll)

res_sm = minimize(nll_softmax, [1.0])
res_lapse = minimize(nll_lapse, [1.0, 0.1], bounds=[(0, None), (0, 1)])
res_bias = minimize(nll_bias, [1.0, 0.0])
res_power = minimize(nll_power, [1.0, 1.0], bounds=[(0, None), (0.1, 5.0)])

print(f"Softmax NLL: {res_sm.fun:.2f}, params: {res_sm.x}")
print(f"Lapse NLL: {res_lapse.fun:.2f}, params: {res_lapse.x}")
print(f"Bias NLL: {res_bias.fun:.2f}, params: {res_bias.x}")
print(f"Power NLL: {res_power.fun:.2f}, params: {res_power.x}")
