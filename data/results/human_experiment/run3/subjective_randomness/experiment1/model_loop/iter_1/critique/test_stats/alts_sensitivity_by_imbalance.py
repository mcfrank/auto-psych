# name: alts_sensitivity_by_imbalance
# description: The difference in correlation between choice and p_alts difference for highly imbalanced vs balanced sequences.
import numpy as np
import pandas as pd

def test_statistic(df):
    p_alts_diff = df['p_alts_a'] - df['p_alts_b']
    high_imb = df[(df['imbalance_a'] > 0.5) & (df['imbalance_b'] > 0.5)].copy()
    low_imb = df[(df['imbalance_a'] <= 0.5) & (df['imbalance_b'] <= 0.5)].copy()
    
    if len(high_imb) < 5 or len(low_imb) < 5: return np.nan
    
    corr_high = high_imb['chose_left'].corr(p_alts_diff.loc[high_imb.index])
    corr_low = low_imb['chose_left'].corr(p_alts_diff.loc[low_imb.index])
    
    val_high = corr_high if not pd.isna(corr_high) else 0.0
    val_low = corr_low if not pd.isna(corr_low) else 0.0
    return float(val_high - val_low)
