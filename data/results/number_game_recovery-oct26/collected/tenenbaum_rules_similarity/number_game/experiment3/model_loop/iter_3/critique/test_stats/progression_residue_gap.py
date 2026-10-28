# name: progression_residue_gap
# description: In sets of >=3 examples whose pairwise differences share a gcd g in 3-12 (g!=10) with nonzero residue (e.g. 41,47,53: 5 mod 6), yes-rate for non-example targets congruent to the examples mod g minus non-congruent targets of the same parity class; observed above null means people pick up 'same step/offset' progressions the model (which only has residue-0 multiples) misses.
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    dist = np.maximum(mn - t, t - mx)
    Xf = np.nan_to_num(X, nan=0.0).astype(np.int64)
    valid = ~np.isnan(X)
    x0 = Xf[:, :1]
    diffs = np.where(valid, np.abs(Xf - x0), 0)
    g = np.gcd.reduce(diffs, axis=1)
    x0 = x0[:, 0]
    ok = (n >= 3) & (g >= 3) & (g <= 12) & (g != 10)
    gs = np.where(ok, g, 1)
    res = np.mod(x0, gs)
    ok &= res != 0
    cong = np.mod(t, gs) == res
    same_par = np.where(np.mod(gs, 2) == 0, np.mod(t, 2) == np.mod(x0, 2), True)
    a = ok & ~is_ex & cong
    b = ok & ~is_ex & ~cong & same_par
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
