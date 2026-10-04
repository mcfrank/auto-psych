# name: balance_preference
# description: Among pairs whose final |#H - #T| differs, the rate of choosing the more balanced sequence; observed above null means the model under-weights head/tail count balance (beyond tally span), below means it over-weights it.
def test_statistic(df):
    a, b = df["sequence_a"], df["sequence_b"]
    ia = (2 * a.str.count("H") - a.str.len()).abs()
    ib = (2 * b.str.count("H") - b.str.len()).abs()
    m = ia != ib
    if not m.any():
        return 0.5
    chose_bal = np.where(ia[m] < ib[m], df["chose_left"][m], 1 - df["chose_left"][m])
    return float(np.mean(chose_bal))
