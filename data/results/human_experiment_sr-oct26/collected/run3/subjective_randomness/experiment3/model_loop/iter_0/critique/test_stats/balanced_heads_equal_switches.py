# name: balanced_heads_equal_switches
# description: Among pairs with equal switch counts but different |#H - #T| imbalance, the rate of choosing the more H/T-balanced sequence; observed above null means people reward face balance beyond switch structure more than the model predicts, below means less.
def _switches(s):
    n = s.str.len().max()
    k = np.zeros(len(s))
    L = s.str.len().to_numpy()
    for i in range(1, n):
        k += (L > i) & (s.str[i - 1].to_numpy() != s.str[i].to_numpy())
    return k
def test_statistic(df):
    sa, sb = df["sequence_a"], df["sequence_b"]
    L = sa.str.len().to_numpy()
    ia = np.abs(2 * sa.str.count("H").to_numpy() - L)
    ib = np.abs(2 * sb.str.count("H").to_numpy() - L)
    m = (_switches(sa) == _switches(sb)) & (ia != ib)
    y = df["chose_left"].to_numpy()
    chose_bal = np.where(ia < ib, y, 1 - y)
    if m.sum() == 0:
        return 0.5
    return float(chose_bal[m].mean())
