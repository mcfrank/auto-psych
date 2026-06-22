# name: extreme_imbalance_penalty
# description: The choice rate for moderately imbalanced sequences (0.1 <= imbalance <= 0.3) over extremely imbalanced sequences (imbalance >= 0.5).
def test_statistic(df):
    a_mod = (df['imbalance_a'] >= 0.1) & (df['imbalance_a'] <= 0.3) & (df['imbalance_b'] >= 0.5)
    b_mod = (df['imbalance_b'] >= 0.1) & (df['imbalance_b'] <= 0.3) & (df['imbalance_a'] >= 0.5)
    valid = df[a_mod | b_mod]
    if len(valid) == 0: return 0.0
    
    chose_mod = (a_mod & (valid['chose_left'] == 1)) | (b_mod & (valid['chose_left'] == 0))
    return float(chose_mod.mean())
