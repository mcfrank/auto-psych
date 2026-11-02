# name: multiple_rule_vs_span_preference
# description: For >=2-example sets whose examples share a common divisor k in 3..12 (largest such k), yes rate on targets that are multiples of k outside the set's span minus yes rate on non-multiples inside the span; observed above null means people favour the multiples rule over magnitude range more than the model, below null means they favour range more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    n = (~np.isnan(ex)).sum(axis=1)
    exi = np.where(np.isnan(ex), 0, ex).astype(np.int64)
    g = np.gcd.reduce(exi, axis=1)
    k = np.zeros(len(g), dtype=np.int64)
    for c in range(3, 13):
        k = np.where((g % c == 0) & (g > 0), c, k)
    t = df["target"].astype(int).to_numpy()
    inr = (t >= np.nanmin(ex, axis=1)) & (t <= np.nanmax(ex, axis=1))
    sel = (n >= 2) & (k > 0)
    mult = (t % np.maximum(k, 1)) == 0
    r = df["rating"].to_numpy().astype(float)
    a = sel & mult & ~inr
    b = sel & ~mult & inr
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
