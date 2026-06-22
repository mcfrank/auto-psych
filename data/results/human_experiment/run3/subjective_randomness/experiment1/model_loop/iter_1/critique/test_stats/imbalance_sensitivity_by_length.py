# name: imbalance_sensitivity_by_length
# description: The difference in choice correlation with imbalance difference between long sequences (n=8) and short sequences (n<=5).
import numpy as np
import pandas as pd

def test_statistic(df):
    imb_diff = df['imbalance_b'] - df['imbalance_a']
    short = df[(df['n_a'] <= 5) & (df['n_b'] <= 5)].copy()
    long_seq = df[(df['n_a'] == 8) & (df['n_b'] == 8)].copy()
    if len(short) < 5 or len(long_seq) < 5: return np.nan
    
    slope_short = short['chose_left'].corr(imb_diff.loc[short.index])
    slope_long = long_seq['chose_left'].corr(imb_diff.loc[long_seq.index])
    
    val_short = slope_short if not pd.isna(slope_short) else 0.0
    val_long = slope_long if not pd.isna(slope_long) else 0.0
    return float(val_long - val_short)
