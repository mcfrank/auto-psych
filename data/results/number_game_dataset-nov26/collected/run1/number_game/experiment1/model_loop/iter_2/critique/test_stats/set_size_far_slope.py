# name: set_size_far_slope
# description: Yes-rate on far targets (>15 from every example, no shared final digit) for 1-example sets minus that for 3-4-example sets; observed > null means people narrow generalization with more examples (size principle) more strongly than summed similarity predicts.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    return ex, t, y

def test_statistic(df):
    ex, t, y = _parse(df)
    valid = ~np.isnan(ex)
    n = valid.sum(1)
    exi = np.nan_to_num(ex, nan=-1).astype(int)
    d = np.nanmin(np.abs(ex - t[:, None]), 1)
    dig = np.any((np.mod(exi, 10) == np.mod(t.astype(int), 10)[:, None]) & valid, 1)
    far = (d > 15) & ~dig
    a = far & (n == 1); b = far & (n >= 3)
    ya = y[a].mean() if a.any() else 0.0
    yb = y[b].mean() if b.any() else 0.0
    return float(ya - yb)
