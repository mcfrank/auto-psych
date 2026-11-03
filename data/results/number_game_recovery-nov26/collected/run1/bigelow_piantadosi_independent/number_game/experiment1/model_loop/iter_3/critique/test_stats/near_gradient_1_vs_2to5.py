# name: near_gradient_1_vs_2to5
# description: Among non-example targets, yes rate at distance exactly 1 from the nearest example minus yes rate at distance 2-5; observed above null means human generalisation is concentrated on immediate neighbours more steeply than the model's fitted neighbourhood width predicts, below null means flatter.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    nearest = dist.min(axis=1)
    nonmember = nearest > 0
    a = r[nonmember & (nearest == 1)]
    b = r[nonmember & (nearest >= 2) & (nearest <= 5)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
