# name: below_edge_near_yes
# description: Yes rate for targets 1-5 below the smallest example; observed above null means the model generalizes too little just below the lower edge of the examples' range (downward extension).
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = np.sum(~np.isnan(X), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(X, axis=1)
    mx = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    m = (mn - t >= 1) & (mn - t <= 5)
    return float(y[m].mean()) if m.sum() > 0 else 0.0
