# name: length_preference
# description: The average choice rate for the longer sequence across all pairs where the two sequences have different lengths.
def test_statistic(df):
    valid = df[df['n_a'] != df['n_b']]
    if len(valid) == 0: return 0.0
    
    chose_longer = ((valid['n_a'] > valid['n_b']) & (valid['chose_left'] == 1)) | \
                   ((valid['n_a'] < valid['n_b']) & (valid['chose_left'] == 0))
    return float(chose_longer.mean())
