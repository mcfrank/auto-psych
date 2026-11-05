# name: within_set_target_sd_small_sets
# description: Mean, over sets with 1-2 examples and >=5 distinct non-example targets, of the SD of per-target yes rates within the set; observed above null means people discriminate between targets of a small set more sharply than the model (below = model's small-set generalization is too all-or-none).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    n = ex.str.len().values
    E = pd.DataFrame(ex.tolist()).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    isex = (E == t[:, None]).any(axis=1)
    mn = np.nanmin(E, axis=1); mx = np.nanmax(E, axis=1)
    d = pd.DataFrame({"s": df["number_set"].values, "t": t, "r": r})[(~isex) & (n <= 2)]
    if len(d) == 0: return 0.0
    c = d.groupby(["s", "t"])["r"].mean().reset_index()
    g = c.groupby("s")["r"].agg(["std", "count"])
    g = g[g["count"] >= 5]
    return float(g["std"].mean()) if len(g) else 0.0
