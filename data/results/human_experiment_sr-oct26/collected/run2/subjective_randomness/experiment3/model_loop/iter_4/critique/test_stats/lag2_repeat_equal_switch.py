# name: lag2_repeat_equal_switch
# description: Among pairs with equal switch counts that differ in the number of 4-flip strictly alternating windows (HTHT/THTH), the rate of choosing the sequence with fewer such windows; observed above null means people see extended alternation as patterned more than the model predicts (model under-penalises local alternation), below means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    seqs = pd.unique(pd.concat([a, b]))
    nsw = {s: sum(x != y for x, y in zip(s, s[1:])) for s in seqs}
    alt = {s: sum(s[i:i + 4] in ("HTHT", "THTH") for i in range(max(0, len(s) - 3))) for s in seqs}
    d = (a.map(alt) - b.map(alt)).to_numpy()
    m = (a.map(nsw) == b.map(nsw)).to_numpy() & (d != 0)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m]
    return float(np.mean(np.where(d[m] < 0, y, 1 - y)))
