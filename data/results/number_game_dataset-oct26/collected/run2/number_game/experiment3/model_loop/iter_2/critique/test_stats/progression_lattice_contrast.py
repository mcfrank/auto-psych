# name: progression_lattice_contrast
# description: For sets of 3+ examples whose pairwise differences share a common step 3<=d<=20 that is neither 10 nor the examples' own common factor (e.g. 53,47,41 -> 5 mod 6), yes-rate on non-example targets on the lattice (t = example mod d) minus off-lattice targets; observed > null means people extrapolate arithmetic-progression lattices that the model's multiples/ends-in-digit concepts miss.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    exv = ex.to_numpy(dtype=float)
    k = ex.notna().sum(axis=1).to_numpy()
    first = exv[:, 0]
    diffs = np.abs(np.nan_to_num(exv - first[:, None], nan=0.0)).astype(int)
    exi = np.nan_to_num(exv, nan=0.0).astype(int)
    d = np.gcd.reduce(diffs, axis=1)
    g = np.gcd.reduce(exi, axis=1)
    ok = (k >= 3) & (d >= 3) & (d <= 20) & (d != 10) & (d != g)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    isex = (exv == t[:, None]).any(axis=1)
    dd = np.where(d > 0, d, 1)
    on = ((t - first.astype(int)) % dd) == 0
    a = ok & ~isex & on
    b = ok & ~isex & ~on
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
