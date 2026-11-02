# name: near_minus_far_distance_gradient
# description: Yes rate on non-example targets within 10 of the nearest example minus yes rate on targets more than 20 from every example; observed above null means people show a steeper numerical-proximity gradient than the model, below means the model's similarity gradient is too steep.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).values
    t = df["target"].astype(float).values
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    r = df["rating"].values.astype(float)
    a, b = (dist >= 1) & (dist <= 10), dist > 20
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
