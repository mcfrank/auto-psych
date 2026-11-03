# name: near_minus_mid_gradient
# description: Yes rate for non-example targets 1-5 away from the nearest example minus yes rate for targets 6-20 away; observed > null means people show a steeper numeric-proximity (similarity) gradient than the model produces, observed < null a flatter one.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    mind = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    r = df["rating"].to_numpy().astype(float)
    a = (mind >= 1) & (mind <= 5)
    b = (mind >= 6) & (mind <= 20)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
