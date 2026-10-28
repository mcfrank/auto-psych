# name: upward_minus_downward_generalization
# description: Yes rate to non-example targets 1-10 above the largest example minus yes rate to targets 1-10 below the smallest example; observed below null means people generalize downward even more (relative to upward) than the upper-edge model predicts, above null means the model's upper-edge asymmetry is too strong.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    mx = np.nanmax(ex, axis=1)
    mn = np.nanmin(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    up = (t > mx) & (t - mx <= 10)
    dn = (t < mn) & (mn - t <= 10)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(y[up].mean() - y[dn].mean())
