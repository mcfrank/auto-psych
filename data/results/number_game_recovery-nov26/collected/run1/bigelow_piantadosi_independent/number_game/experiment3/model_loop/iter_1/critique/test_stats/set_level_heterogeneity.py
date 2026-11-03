# name: set_level_heterogeneity
# description: Standard deviation, across example sets with at least 20 non-example trials, of each set's non-example yes-rate; observed above the null means sets differ more in overall generalization breadth (some diffuse, some tight) than the model reproduces, below means the model exaggerates between-set differences.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    Dg = np.where(np.isnan(E), np.nan, np.mod(E, 10))
    is_ex = np.any(E == t[:, None], axis=1)
    g = pd.DataFrame({"s": df["number_set"].astype(str).to_numpy()[~is_ex], "r": r[~is_ex]}).groupby("s")["r"].agg(["mean", "size"])
    g = g[g["size"] >= 20]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
