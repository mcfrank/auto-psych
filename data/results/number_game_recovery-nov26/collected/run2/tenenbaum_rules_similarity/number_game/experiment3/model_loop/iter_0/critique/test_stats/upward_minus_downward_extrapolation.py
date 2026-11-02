# name: upward_minus_downward_extrapolation
# description: Yes rate on targets 1-15 above the largest example minus yes rate on targets 1-15 below the smallest example (sets of 2+ examples); observed above null means people extend ranges upward more (or downward less) than the model's symmetric interval/rule account predicts.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).values
    t = df["target"].astype(float).values
    n = (~np.isnan(ex)).sum(axis=1)
    mx, mn = np.nanmax(ex, axis=1), np.nanmin(ex, axis=1)
    r = df["rating"].values.astype(float)
    up = (n >= 2) & (t > mx) & (t <= mx + 15)
    dn = (n >= 2) & (t < mn) & (t >= mn - 15)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(r[up].mean() - r[dn].mean())
