# name: near_vs_mid_distance_gradient
# description: Yes rate for non-example targets within 3 of the nearest example minus yes rate for targets 16-30 away; observed below the null means people's generalization falls off with distance more shallowly than the model's interval/rule mixture predicts (above = steeper).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    t = df["target"].astype(float).values[:, None]
    d = np.nanmin(np.abs(ex - t), axis=1)
    r = df["rating"].values.astype(float)
    a = (d >= 1) & (d <= 3)
    b = (d >= 16) & (d <= 30)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
