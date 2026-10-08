# name: balance_pref_equal_switches
# description: Among pairs with equal switch counts but different |#H - #T|, the rate of choosing the more balanced sequence; observed above null_mean means people reward H/T balance more than the model's biased-coin generator implies (below: less).
def test_statistic(df):
    def sw(s):
        return sum(x != y for x, y in zip(s, s[1:]))
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    SW = {s: sw(s) for s in u}
    ka = df["sequence_a"].map(SW).values; kb = df["sequence_b"].map(SW).values
    L = df["sequence_a"].str.len().values
    ia = np.abs(2 * df["sequence_a"].str.count("H").values - L)
    ib = np.abs(2 * df["sequence_b"].str.count("H").values - L)
    m = (ka == kb) & (ia != ib)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].values
    return float(np.mean(np.where(ia < ib, y, 1 - y)[m]))
