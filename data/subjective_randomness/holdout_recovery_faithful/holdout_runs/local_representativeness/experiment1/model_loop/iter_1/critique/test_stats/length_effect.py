# name: length_effect
# description: The slope of the choice probability on the difference in length (n_a - n_b).
import numpy as np
def test_statistic(df):
    mask = df['n_a'] != df['n_b']
    if not mask.any(): return 0.0
    x = (df.loc[mask, 'n_a'] - df.loc[mask, 'n_b']).values
    y = df.loc[mask, 'chose_left'].values
    if len(np.unique(x)) < 2: return 0.0
    return float(np.polyfit(x, y, 1)[0])
