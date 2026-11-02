# name: cell_unanimity
# description: Fraction of (set, target) cells with >=2 responses in which all responses agree; observed above null means responses are more deterministic/extreme per item than the model's graded posterior predicts, below null means more disagreement.
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
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    g = pd.DataFrame({"k": key.to_numpy(), "r": r}).groupby("k")["r"].agg(["mean", "count"])
    g = g[g["count"] >= 2]
    if len(g) == 0:
        return 0.0
    return float(((g["mean"] == 0) | (g["mean"] == 1)).mean())
