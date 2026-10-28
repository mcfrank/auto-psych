# name: parity_rule_gap_far
# description: Among sets of 3-4 examples that all share parity but not a final digit, yes rate for same-parity targets minus opposite-parity targets, restricted to targets >5 from every example and sharing no final digit with any; observed > model means people extract an odd/even rule more strongly than coherence-stretched parity attention predicts (observed < model: the model over-weights parity).
def test_statistic(df):
    s = df["number_set"].astype(str)
    ex = s.str.split(",", expand=True).reindex(columns=range(4))
    ex = ex.apply(pd.to_numeric, errors="coerce").values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    valid = ~np.isnan(ex)
    k = valid.sum(1)
    e = np.nan_to_num(ex)
    par = np.where(valid, e % 2, np.nan)
    allpar = np.nanmax(par, 1) == np.nanmin(par, 1)
    dg = np.where(valid, e % 10, np.nan)
    alldig = np.nanmax(dg, 1) == np.nanmin(dg, 1)
    d = np.where(valid, np.abs(t[:, None] - ex), np.inf)
    match = (valid & (e % 10 == (t % 10)[:, None])).any(1)
    base = (k >= 3) & allpar & ~alldig & (d.min(1) > 5) & ~match
    same = (t % 2) == np.nanmin(par, 1)
    a = base & same
    b = base & ~same
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
