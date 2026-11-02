# name: multi_set_nearest_example_gradient
# description: Among sets with 2+ examples and targets not in the set, yes rate for targets 1-3 away from the nearest example minus yes rate for targets 4-15 away; observed above null means people show a sharper local similarity gradient around individual examples than the model's ranges and rules produce, below null means a flatter one.
def test_statistic(df):
    s = df["number_set"].astype(str)
    parts = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    d = np.abs(parts - t[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    dmin = d.min(axis=1)
    multi = (s.str.count(",") >= 1).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    near = multi & (dmin >= 1) & (dmin <= 3)
    mid = multi & (dmin >= 4) & (dmin <= 15)
    if near.sum() == 0 or mid.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[mid].mean())
