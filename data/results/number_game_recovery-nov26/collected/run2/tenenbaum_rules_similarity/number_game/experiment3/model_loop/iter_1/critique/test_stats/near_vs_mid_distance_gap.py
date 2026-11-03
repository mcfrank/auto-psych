# name: near_vs_mid_distance_gap
# description: Yes-rate for non-example targets 1-2 from the nearest example minus yes-rate for targets 5-10 away; observed below null means people do not show the steep near-neighbour similarity gradient the model predicts (and vice versa).
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    dist = np.where(~np.isnan(ex), np.abs(ex - t[:, None]), np.inf).min(axis=1)
    a = (dist >= 1) & (dist <= 2)
    b = (dist >= 5) & (dist <= 10)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
