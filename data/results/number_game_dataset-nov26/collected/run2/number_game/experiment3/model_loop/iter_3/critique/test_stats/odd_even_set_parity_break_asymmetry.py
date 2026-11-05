# name: odd_even_set_parity_break_asymmetry
# description: Among multi-example sets of uniform parity, yes rate for even targets after all-odd sets minus yes rate for odd targets after all-even sets; nonzero observed vs null means parity violations are tolerated asymmetrically (odd vs even sets) in a way the model's symmetric parity treatment misses.
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    u, inv = np.unique(sets, return_inverse=True)
    par = np.full(len(u), -1)
    for i, s in enumerate(u):
        ex = [int(x) for x in s.split(",")]
        ps = {e % 2 for e in ex}
        if len(ex) >= 2 and len(ps) == 1:
            par[i] = ps.pop()
    p = par[inv]
    t = df["target"].astype(int).values % 2
    r = df["rating"].astype(float).values
    a = (p == 1) & (t == 0)
    b = (p == 0) & (t == 1)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
