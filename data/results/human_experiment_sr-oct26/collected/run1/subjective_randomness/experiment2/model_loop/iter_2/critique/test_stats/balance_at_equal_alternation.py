# name: balance_at_equal_alternation
# description: Among trials where both sequences have the same number of alternations but different |#H - #T|, the proportion choosing the MORE balanced sequence; observed above the null means people weight heads/tails balance more than the model's imbalance term predicts once switching is held fixed, below means less.
def test_statistic(df):
    def alt(s):
        return sum(1 for x, y in zip(s, s[1:]) if x != y)
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    ma = {s: alt(s) for s in u}
    mi = {s: abs(s.count("H") - s.count("T")) for s in u}
    aa = df["sequence_a"].map(ma).to_numpy()
    ab = df["sequence_b"].map(ma).to_numpy()
    ia = df["sequence_a"].map(mi).to_numpy()
    ib = df["sequence_b"].map(mi).to_numpy()
    sel = (aa == ab) & (ia != ib)
    if sel.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    chose = np.where(ia < ib, c, 1 - c)
    return float(chose[sel].mean())
