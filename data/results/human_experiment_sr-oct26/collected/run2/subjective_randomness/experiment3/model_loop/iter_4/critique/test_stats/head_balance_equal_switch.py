# name: head_balance_equal_switch
# description: Among pairs with equal switch counts whose sequences differ in |#H - #T|, the rate of choosing the more balanced sequence; observed above null means people value an even H/T count more than the model's tally-span cue predicts (model under-produces), below means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    seqs = pd.unique(pd.concat([a, b]))
    nsw = {s: sum(x != y for x, y in zip(s, s[1:])) for s in seqs}
    imb = {s: abs(2 * s.count("H") - len(s)) for s in seqs}
    d = (a.map(imb) - b.map(imb)).to_numpy()
    m = (a.map(nsw) == b.map(nsw)).to_numpy() & (d != 0)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m]
    return float(np.mean(np.where(d[m] < 0, y, 1 - y)))
