# name: length_preference_for_balanced
# description: The choice rate for the longer sequence when both sequences are well-balanced (imbalance <= 0.25).
def test_statistic(df):
    subset = df[(df['imbalance_a'] <= 0.25) & (df['imbalance_b'] <= 0.25)]
    valid = subset[subset['n_a'] != subset['n_b']]
    if len(valid) == 0: return 0.0
    
    chose_longer = ((valid['n_a'] > valid['n_b']) & (valid['chose_left'] == 1)) | \
                   ((valid['n_a'] < valid['n_b']) & (valid['chose_left'] == 0))
    return float(chose_longer.mean())
