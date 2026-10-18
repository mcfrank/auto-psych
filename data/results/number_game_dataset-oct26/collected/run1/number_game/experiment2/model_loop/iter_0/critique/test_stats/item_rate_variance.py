# name: item_rate_variance
# description: Variance across (set,target) items of the per-item yes-rate; observed above model mean means items differ more systematically than the model's three features can explain (unexplained stimulus structure).
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(X)
    n = valid.sum(axis=1)
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    m = pd.Series(r).groupby(key.to_numpy()).mean()
    return float(m.var(ddof=0))
