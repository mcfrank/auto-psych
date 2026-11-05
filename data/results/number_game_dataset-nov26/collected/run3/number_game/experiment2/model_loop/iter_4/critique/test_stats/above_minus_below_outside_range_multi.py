# name: above_minus_below_outside_range_multi
# description: For sets of 2+ examples, yes rate on targets above the largest example minus yes rate on targets below the smallest example; observed above null means the model over-predicts downward extrapolation relative to upward.
def test_statistic(df):
    s = df["number_set"].astype(str)
    sp = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = sp.notna().sum(axis=1).to_numpy()
    lo = sp.min(axis=1).to_numpy()
    hi = sp.max(axis=1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    a = (n >= 2) & (t > hi)
    b = (n >= 2) & (t < lo)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
