# name: set_level_yes_rate_sd
# description: Standard deviation, across example sets with >=20 non-example trials, of the set's yes-rate on non-example targets; observed above null_mean means human generalisation differs between sets more than the model predicts (model too uniform across sets), below means the model exaggerates set-to-set differences.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    isex = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    dmin = np.nanmin(np.abs(E - t[:, None]), axis=1)
    ld = np.mod(E, 10)
    d = pd.DataFrame({"s": df["number_set"].astype(str).to_numpy(), "r": r})[~isex]
    g = d.groupby("s")["r"].agg(["mean", "count"])
    g = g[g["count"] >= 20]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
