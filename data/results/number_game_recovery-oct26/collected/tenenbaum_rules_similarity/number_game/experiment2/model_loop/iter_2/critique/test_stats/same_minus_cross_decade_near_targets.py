# name: same_minus_cross_decade_near_targets
# description: Among non-example targets 1-5 away from the nearest example, yes rate when the target lies in the same decade (same tens digit, 1-10 counted as 0-9 style via (x-1)//10) as that nearest example minus when it lies in a different decade; observed above null means people respect decade boundaries more strongly than the model's decade bonus, below null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    mx = np.nanmax(ex, axis=1)
    mn = np.nanmin(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    idx = np.nanargmin(np.abs(ex - t[:, None]), axis=1)
    near = ex[np.arange(len(ex)), idx]
    m = ~is_ex & (dist >= 1) & (dist <= 5)
    same = ((t - 1) // 10) == ((near - 1) // 10)
    a = m & same; b = m & ~same
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
