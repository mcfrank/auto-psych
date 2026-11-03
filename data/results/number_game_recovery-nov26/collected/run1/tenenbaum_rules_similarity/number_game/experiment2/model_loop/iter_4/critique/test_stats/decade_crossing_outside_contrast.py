# name: decade_crossing_outside_contrast
# description: Among non-member targets 1-9 outside the span of multi-example sets, yes rate for targets in the same decade (same tens digit, 100 counted with 90s) as the nearest edge example minus yes rate for targets in a different decade; observed above null means people treat decades as category boundaries more than the model predicts, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    member = np.any(ex == t[:, None], axis=1)
    d = np.where(t > hi, t - hi, np.where(t < lo, lo - t, 0))
    edge = np.where(t > hi, hi, lo)
    dec = lambda x: np.minimum((x - 1) // 10, 9)
    sel = (k >= 2) & (d >= 1) & (d <= 9) & ~member
    same = dec(t) == dec(edge)
    a = sel & same; b = sel & ~same
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
