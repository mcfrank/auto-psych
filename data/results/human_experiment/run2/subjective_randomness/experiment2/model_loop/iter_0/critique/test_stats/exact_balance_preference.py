# name: exact_balance_preference
# description: The choice rate for exactly balanced sequences (imbalance=0) over slightly imbalanced ones (0 < imbalance <= 0.3).
def test_statistic(df):
    a_bal = (df['imbalance_a'] == 0) & (df['imbalance_b'] > 0) & (df['imbalance_b'] <= 0.3)
    b_bal = (df['imbalance_b'] == 0) & (df['imbalance_a'] > 0) & (df['imbalance_a'] <= 0.3)
    valid = df[a_bal | b_bal]
    if len(valid) == 0: return 0.0
    
    chose_bal = (a_bal & (valid['chose_left'] == 1)) | (b_bal & (valid['chose_left'] == 0))
    return float(chose_bal.mean())
