# name: alt_rate_diff_choice_correlation
# description: Pearson correlation between (p_alts_b - p_alts_a) and chose_left across all trials; tests the sign and slope of the alternation-rate contrast on choice, probing whether the model's alternation-penalization is directionally correct.
def test_statistic(df):
    alt_diff = df['p_alts_b'] - df['p_alts_a']
    r = np.corrcoef(alt_diff.values, df['chose_left'].values)[0, 1]
    return float(r)
