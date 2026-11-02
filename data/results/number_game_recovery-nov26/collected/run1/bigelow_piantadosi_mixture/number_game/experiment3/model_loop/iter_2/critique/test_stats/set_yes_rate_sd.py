# name: set_yes_rate_sd
# description: Standard deviation across sets (with >=20 trials) of each set's overall yes rate; observed > null means sets differ in generalisation breadth more than the model predicts (model too homogeneous across sets), observed < null less.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    absd = np.abs(E - t[:, None])
    d = np.nanmin(absd, axis=1)
    ne = E[np.arange(len(t)), np.nanargmin(absd, axis=1)]
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    g = pd.DataFrame({"s": df["number_set"].astype(str).to_numpy(), "y": y}).groupby("s")["y"].agg(["mean", "size"])
    g = g[g["size"] >= 20]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
