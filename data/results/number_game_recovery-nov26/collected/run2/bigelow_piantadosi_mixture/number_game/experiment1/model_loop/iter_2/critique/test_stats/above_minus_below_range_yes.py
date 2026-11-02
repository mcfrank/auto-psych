# name: above_minus_below_range_yes
# description: Yes rate for non-digit-sharing targets 1-20 above the example maximum minus that for targets 1-20 below the example minimum; observed differing from the null means the model's symmetric intervals miss a directional (upward vs downward) extension bias, positive = humans extend upward more than predicted.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    td = np.array([[str(d) in str(v) for d in range(10)] for v in range(101)])
    exi = np.nan_to_num(ex, nan=0).astype(int)
    exd = np.zeros((len(df), 10), bool)
    for j in range(ex.shape[1]):
        exd |= td[exi[:, j]] & ~np.isnan(ex[:, j])[:, None]
    share = np.any(exd & td[t], axis=1)
    up = (t > hi) & (t <= hi + 20) & ~share
    dn = (t < lo) & (t >= lo - 20) & ~share
    a = y[up].mean() if up.any() else 0.0
    b = y[dn].mean() if dn.any() else 0.0
    return float(a - b)
