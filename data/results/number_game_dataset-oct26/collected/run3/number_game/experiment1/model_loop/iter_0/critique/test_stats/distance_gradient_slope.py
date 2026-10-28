# name: distance_gradient_slope
# description: OLS slope of yes on log(1 + distance from target to its nearest example), after subtracting each set's mean yes rate; more negative observed than null_mean means human generalization falls off with numeric distance more steeply than the model predicts, less negative means flatter.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()[:, None].astype(float)
    d = np.nanmin(np.abs(s - t), axis=1)
    x = np.log1p(d)
    y = df["rating"].astype(float).to_numpy()
    g = df["number_set"].astype(str)
    y = y - pd.Series(y).groupby(g.to_numpy()).transform("mean").to_numpy()
    x = x - pd.Series(x).groupby(g.to_numpy()).transform("mean").to_numpy()
    v = (x * x).sum()
    if v == 0:
        return 0.0
    return float((x * y).sum() / v)
