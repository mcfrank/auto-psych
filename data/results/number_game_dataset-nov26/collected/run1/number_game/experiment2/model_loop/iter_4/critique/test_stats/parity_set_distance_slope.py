# name: parity_set_distance_slope
# description: In sets of 3+ examples all of one parity but not all sharing a final digit, OLS slope of yes on distance-to-nearest-example (per 10 units) among same-parity targets sharing no example's final digit; observed > null (flatter) means people generalize a parity rule uniformly across the range, observed < null means a steeper magnitude gradient than the model predicts.
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
    p0 = np.mod(e1[:, 0], 2)
    par = (~pres | (np.mod(e1, 2) == p0[:, None])).all(1) & (n >= 3)
    u0 = np.mod(e1[:, 0], 10)
    run = (~pres | (np.mod(e1, 10) == u0[:, None])).all(1)
    a = par & ~run & (np.mod(t1, 2) == p0) & ~digmatch
    if a.sum() < 3:
        return 0.0
    x = d[a] / 10.0
    y = r[a]
    vx = np.var(x)
    if vx <= 0:
        return 0.0
    return float(np.mean((x - x.mean()) * (y - y.mean())) / vx)
