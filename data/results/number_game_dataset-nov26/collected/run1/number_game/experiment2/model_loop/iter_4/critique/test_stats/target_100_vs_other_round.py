# name: target_100_vs_other_round
# description: Over sets not containing the target, yes-rate for target 100 minus yes-rate for targets 10-90 that are multiples of 10; observed > null means people give 100 a special landmark/endpoint status beyond the model's generic round-number, magnitude and divisor similarity, observed < null means the model over-credits 100.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t1 = df["target"].astype(int).to_numpy().astype(float)
    t = t1[:, None]
    pres = ~np.isnan(ex)
    n = pres.sum(1)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    r = df["rating"].to_numpy(dtype=float)
    d = np.abs(t - e).min(1)
    digmatch = ((np.mod(t, 10) == np.mod(e1, 10)) & pres).any(1)
    def diff(a, b):
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return float(r[a].mean() - r[b].mean())
    isex = ((t == e) & pres).any(1)
    a = (t1 == 100) & ~isex
    b = (np.mod(t1, 10) == 0) & (t1 < 100) & ~isex
    return diff(a, b)
