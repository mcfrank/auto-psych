# name: near_vs_mid_distance_gradient
# description: Yes rate for targets 1-3 away from the nearest example minus yes rate for targets 8-15 away (targets not in the set); observed < null means people's similarity gradient is flatter than the model's interval-driven gradient (model overpredicts yes for close neighbours relative to moderately distant ones).
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    r = df["rating"].to_numpy().astype(float)
    near = (dist >= 1) & (dist <= 3)
    mid = (dist >= 8) & (dist <= 15)
    if near.sum() == 0 or mid.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[mid].mean())
