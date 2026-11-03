# name: setsize_nonexample_yes_slope
# description: OLS slope of yes (0/1) on number of examples (1-4) among non-example targets; observed more negative than null means people narrow their generalisation with more examples faster than the size principle predicts, less negative/positive means slower.
def _ex(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = (~np.isnan(E)).sum(1)
    isex = (E == t[:, None]).any(1)
    return E, t, r, n, isex

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    E, t, r, n, isex = _ex(df)
    m = ~isex
    x = n[m].astype(float); y = r[m]
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / max((xc ** 2).sum(), 1e-9))
