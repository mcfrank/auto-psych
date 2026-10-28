# name: magnitude_scaled_generalization
# description: Yes rate for targets 1-10 outside the examples' range when all examples are >= 50, minus the same when all examples are <= 30; the model's intervals are linear and location-free, so observed above null_mean means people generalize further around large numbers (log-like magnitude compression) than the model does.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",")
    ex = pd.DataFrame(sets.tolist(), index=df.index).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    n_ex = np.sum(~np.isnan(ex), axis=1)
    outside = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    m = (outside >= 1) & (outside <= 10)
    a = m & (lo >= 50); b = m & (hi <= 30)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
