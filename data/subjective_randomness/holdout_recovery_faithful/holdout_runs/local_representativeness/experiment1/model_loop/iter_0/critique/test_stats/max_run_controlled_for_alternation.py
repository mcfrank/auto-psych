# name: max_run_controlled_for_alternation
# description: Correlation between the difference in max_run and choice, computed only on trials where sequences have similar alternation rates, testing if max_run is a missing cue.
import numpy as np
import pandas as pd

def test_statistic(df):
    mask = abs(df['p_alts_a'] - df['p_alts_b']) <= 0.2
    if mask.sum() < 5: return 0.0
    sub = df[mask]
    diff = sub['max_run_a'] - sub['max_run_b']
    if diff.std() == 0 or sub['chose_left'].std() == 0: return 0.0
    return float(np.corrcoef(diff, sub['chose_left'])[0, 1])
