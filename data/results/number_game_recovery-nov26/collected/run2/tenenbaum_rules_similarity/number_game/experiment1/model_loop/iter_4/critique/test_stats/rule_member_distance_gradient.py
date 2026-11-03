# name: rule_member_distance_gradient
# description: For multi-example sets whose examples share a common divisor >= 3, among targets that are multiples of that divisor (not in the set), yes-rate for targets within 10 of the nearest example minus those more than 20 away; observed above null means people show a numeric-similarity gradient even among rule-consistent targets that the model's flat rule extension misses, below null the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(ex)
    exi = np.where(valid, ex, 0).astype(np.int64)
    g = np.gcd.reduce(exi, axis=1)
    nex = valid.sum(axis=1)
    t = df["target"].astype(int).to_numpy()
    inset = (ex == t[:, None]).any(axis=1)
    member = (g >= 3) & (nex >= 2) & ~inset & (t % np.maximum(g, 1) == 0)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    r = df["rating"].to_numpy(dtype=float)
    a = member & (d <= 10)
    b = member & (d > 20)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
