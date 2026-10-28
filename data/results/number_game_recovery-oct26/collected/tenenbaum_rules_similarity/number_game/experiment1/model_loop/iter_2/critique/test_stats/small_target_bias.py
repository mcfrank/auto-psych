# name: small_target_bias
# description: Among non-example targets more than 10 away from every example, yes rate for targets <= 20 minus targets >= 80; the model has no target-magnitude preference, so observed above null_mean means people favour small numbers as program outputs beyond what the model's concepts produce (below: favour large numbers).
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
    m = (~member) & (dist > 10)
    a = m & (t <= 20); b = m & (t >= 80)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
