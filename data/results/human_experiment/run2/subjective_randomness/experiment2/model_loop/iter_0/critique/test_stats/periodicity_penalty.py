# name: periodicity_penalty
# description: The choice rate for the less periodic sequence, conditioned on the sequences having similar alternation rates and imbalances.
def test_statistic(df):
    subset = df[(abs(df['p_alts_a'] - df['p_alts_b']) < 0.15) & 
                (abs(df['imbalance_a'] - df['imbalance_b']) < 0.15)]
    valid = subset[subset['periodicity_a'] != subset['periodicity_b']]
    if len(valid) == 0: return 0.0
    
    chose_less_periodic = ((valid['periodicity_a'] < valid['periodicity_b']) & (valid['chose_left'] == 1)) | \
                          ((valid['periodicity_a'] > valid['periodicity_b']) & (valid['chose_left'] == 0))
    return float(chose_less_periodic.mean())
