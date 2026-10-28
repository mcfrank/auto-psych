# name: round_number_target_boost
# description: For sets containing no multiple of 10, among non-example targets, yes rate for targets that are multiples of 10 minus other targets; the model gives multiples-of-10 rules ~zero posterior for such sets, so observed above null_mean means people endorse round numbers more than the model, below means less.
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
    no_round = ~np.any(np.mod(ex, 10) == 0, axis=1)
    m = no_round & (~member)
    r = np.mod(t, 10) == 0
    a = m & r; b = m & ~r
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
