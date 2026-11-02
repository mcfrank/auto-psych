# name: above_minus_below_near
# description: Yes rate on targets 1-10 above the largest example minus yes rate on targets 1-10 below the smallest example (sets of 2+ examples); observed above null means people favour upward extension of ranges more than the model's upward-room term predicts, below means less.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    nd = x.sub(t, axis=0).abs().min(axis=1)
    up = (t > mx) & (t - mx <= 10) & (n >= 2)
    dn = (t < mn) & (mn - t <= 10) & (n >= 2)
    v = r[up].mean() - r[dn].mean()
    return float(0.0 if (v is None or not np.isfinite(v)) else v)
