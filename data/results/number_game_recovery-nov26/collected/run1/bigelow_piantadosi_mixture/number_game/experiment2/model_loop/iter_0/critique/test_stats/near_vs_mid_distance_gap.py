# name: near_vs_mid_distance_gap
# description: Yes rate for non-example targets 1-5 from the nearest example minus yes rate for targets 6-20 away; observed above null_mean means people show a steeper local similarity gradient than the model's interval/rule mixture (model's generalisation too flat or non-monotone near examples).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    n_ex = np.sum(~np.isnan(E), axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    a = (dist >= 1) & (dist <= 5)
    b = (dist >= 6) & (dist <= 20)
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
