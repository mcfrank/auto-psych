# name: same_decade_effect
# description: Among non-example targets 1-9 away from the nearest example, yes rate when the target shares its tens digit (x//10) with some example minus when it does not; the model has no decade concepts, so observed above null_mean means people use decade membership more than the model, below means less.
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
    same = np.any(np.floor(ex / 10) == np.floor(t / 10)[:, None], axis=1)
    m = (~member) & (dist >= 1) & (dist <= 9)
    a = m & same; b = m & ~same
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
