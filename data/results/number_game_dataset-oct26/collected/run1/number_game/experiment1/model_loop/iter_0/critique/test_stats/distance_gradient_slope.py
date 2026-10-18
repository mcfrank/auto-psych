# name: distance_gradient_slope
# description: OLS slope of yes on distance (in tens) from target to nearest example, among targets not in the set; observed more negative than null means humans show a steeper numeric-similarity gradient than the model, more positive means a flatter (or reversed) one.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4))
    ex = ex.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    return ex, t, r

def test_statistic(df):
    ex, t, r = _parse(df)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = d > 0
    x, y = d[m] / 10.0, r[m]
    if m.sum() < 3 or x.std() == 0:
        return 0.0
    return float(np.cov(x, y, bias=True)[0, 1] / x.var())
