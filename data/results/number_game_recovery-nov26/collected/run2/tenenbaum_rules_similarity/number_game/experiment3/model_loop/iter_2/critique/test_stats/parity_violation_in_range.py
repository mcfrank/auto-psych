# name: parity_violation_in_range
# description: Among sets of 3+ examples that all share parity, yes rate for opposite-parity targets strictly inside the example range; observed above null means people fill in the range ignoring parity more than the model predicts (below null: they respect the parity rule more strongly than the model).
def test_statistic(df):
    parts = df["number_set"].str.split(",", expand=True).astype(float)
    n = parts.notna().sum(axis=1).values
    vals = parts.values
    p = np.mod(vals, 2)
    p0 = p[:, 0]
    same = np.all((p == p0[:, None]) | np.isnan(vals), axis=1)
    mx = np.nanmax(vals, axis=1)
    mn = np.nanmin(vals, axis=1)
    t = df["target"].astype(float).values
    m = (n >= 3) & same & (np.mod(t, 2) != p0) & (t > mn) & (t < mx)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
