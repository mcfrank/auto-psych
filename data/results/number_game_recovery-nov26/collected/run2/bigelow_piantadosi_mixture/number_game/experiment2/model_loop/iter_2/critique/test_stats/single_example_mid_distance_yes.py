# name: single_example_mid_distance_yes
# description: Yes rate for single-example sets at targets 6-20 away from the example; observed below null means human generalization falls off with distance more steeply than the model predicts (observed above null: more shallowly).
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = np.sum(~np.isnan(X), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(X, axis=1)
    mx = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    m = (n == 1) & (dist >= 6) & (dist <= 20)
    return float(y[m].mean()) if m.sum() > 0 else 0.0
