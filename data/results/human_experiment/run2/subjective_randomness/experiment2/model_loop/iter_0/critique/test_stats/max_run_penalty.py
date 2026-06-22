# name: max_run_penalty
# description: Choice rate for the sequence with the shorter maximum run length, conditioned on the sequences having similar alternation rates.
def test_statistic(df):
    subset = df[abs(df['p_alts_a'] - df['p_alts_b']) <= 0.1]
    valid = subset[subset['max_run_a'] != subset['max_run_b']]
    if len(valid) == 0: return 0.0
    
    chose_smaller_run = ((valid['max_run_a'] < valid['max_run_b']) & (valid['chose_left'] == 1)) | \
                        ((valid['max_run_a'] > valid['max_run_b']) & (valid['chose_left'] == 0))
    return float(chose_smaller_run.mean())
