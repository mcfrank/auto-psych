# name: local_imbalance_controlled_for_global
# description: Correlation between difference in local imbalance and choice on trials where global imbalance is similar, testing if the model misses local representativeness.
import numpy as np
import pandas as pd

def test_statistic(df):
    mask = abs(df['imbalance_a'] - df['imbalance_b']) <= 0.2
    if mask.sum() < 5: return 0.0
    sub = df[mask]
    diff = sub['local_imbalance_a'] - sub['local_imbalance_b']
    if diff.std() == 0 or sub['chose_left'].std() == 0: return 0.0
    return float(np.corrcoef(diff, sub['chose_left'])[0, 1])
