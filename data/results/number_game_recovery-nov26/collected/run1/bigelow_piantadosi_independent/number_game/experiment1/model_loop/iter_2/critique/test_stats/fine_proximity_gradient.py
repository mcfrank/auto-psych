# name: fine_proximity_gradient
# description: Among non-example targets within 10 of the nearest example, yes rate at distance 1-3 minus at distance 4-10; observed above null means human generalisation decays over short numerical distances more steeply than the model's interval hypotheses predict, below null means flatter.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    nearest = dist.min(axis=1)
    nonmember = nearest > 0
    a = r[nonmember & (nearest <= 3)]
    b = r[nonmember & (nearest >= 4) & (nearest <= 10)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
