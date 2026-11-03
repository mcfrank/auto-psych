# name: single_example_near_far_gradient
# description: For one-example sets, yes rate for targets 1-5 from the example minus yes rate for targets more than 20 away; observed > null means people's similarity generalisation from a lone example is steeper than the model's, observed < null shallower.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    a = (n == 1) & (d >= 1) & (d <= 5)
    b = (n == 1) & (d > 20)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
