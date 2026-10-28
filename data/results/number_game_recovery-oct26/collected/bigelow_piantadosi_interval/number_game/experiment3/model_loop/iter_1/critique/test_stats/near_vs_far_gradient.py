# name: near_vs_far_gradient
# description: Among non-example targets, yes rate for targets within 3 of the nearest example minus yes rate for targets 20+ away from every example; observed above null_mean means people show a steeper numeric-proximity (interval) generalization gradient than the model, below means a flatter one.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()[:, None]
    dist = np.nanmin(np.abs(ex - t), axis=1)
    r = df["rating"].to_numpy().astype(float)
    near = (dist >= 1) & (dist <= 3)
    far = dist >= 20
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
