# name: set_breadth_spread
# description: Standard deviation across example sets (with >=20 responses) of each set's mean yes rate; observed above null means sets differ in how broadly people generalise more than the model predicts (some near-zero, others very broad), below null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    nearest = dist.min(axis=1)
    nonmember = nearest > 0
    lo = np.where(valid, ex, np.inf).min(axis=1)
    hi = np.where(valid, ex, -np.inf).max(axis=1)
    g = pd.DataFrame({"s": df["number_set"].astype(str).to_numpy(), "r": r}).groupby("s")["r"].agg(["mean", "count"])
    g = g[g["count"] >= 20]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
