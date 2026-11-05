# name: near_gradient_steepness
# description: Yes-rate for targets 1-3 from nearest example minus yes-rate for targets 8-12 away (no shared final digit in both bins); observed > null means people's magnitude generalization gradient falls off more steeply near examples than the model predicts, observed < null flatter.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    return ex, t, y

def test_statistic(df):
    ex, t, y = _parse(df)
    valid = ~np.isnan(ex)
    exi = np.nan_to_num(ex, nan=-1).astype(int)
    d = np.nanmin(np.abs(ex - t[:, None]), 1)
    dig = np.any((np.mod(exi, 10) == np.mod(t.astype(int), 10)[:, None]) & valid, 1)
    a = (d >= 1) & (d <= 3) & ~dig; b = (d >= 8) & (d <= 12) & ~dig
    ya = y[a].mean() if a.any() else 0.0
    yb = y[b].mean() if b.any() else 0.0
    return float(ya - yb)
