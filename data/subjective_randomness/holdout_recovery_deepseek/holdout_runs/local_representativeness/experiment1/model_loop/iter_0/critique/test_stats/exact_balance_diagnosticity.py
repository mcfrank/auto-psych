# name: exact_balance_diagnosticity
# description: Net probability of choosing the left sequence when the left has an exactly equal head count (p==0.5) rather than the right, testing whether humans conflate exact symmetry with contrivance beyond the proportion itself.
def test_statistic(df):
    eq_a = np.isclose(df['p_a'].astype(float), 0.5)
    eq_b = np.isclose(df['p_b'].astype(float), 0.5)
    a = df.loc[eq_a & ~eq_b]
    b = df.loc[eq_b & ~eq_a]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return a['chose_left'].mean() - b['chose_left'].mean()
