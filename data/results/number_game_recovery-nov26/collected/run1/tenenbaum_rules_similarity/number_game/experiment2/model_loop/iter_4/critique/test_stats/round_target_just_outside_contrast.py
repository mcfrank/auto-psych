# name: round_target_just_outside_contrast
# description: Among non-member targets 1-10 outside the example span of multi-example sets, yes rate for multiples of 10 minus yes rate for other targets; observed above null means people extend ranges to round numbers more than the round-edge prior predicts, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    member = np.any(ex == t[:, None], axis=1)
    d = np.where(t > hi, t - hi, np.where(t < lo, lo - t, 0))
    sel = (k >= 2) & (d >= 1) & (d <= 10) & ~member
    rnd = (t % 10 == 0)
    a = sel & rnd; b = sel & ~rnd
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
