# name: outside_distance_slope
# description: For targets outside the examples' [min,max] range, OLS slope of yes (0/1) on log(1 + distance to the nearest example); a more negative observed than null means people's generalization falls off with distance more steeply than the model predicts (less negative = flatter).
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric)
    t = df["target"].astype(int)
    mn, mx = s.min(axis=1), s.max(axis=1)
    out = (t < mn) | (t > mx)
    d = s.sub(t, axis=0).abs().min(axis=1)
    x = np.log1p(d[out].to_numpy(float))
    y = df["rating"][out].to_numpy(float)
    if len(x) < 3 or np.var(x) == 0:
        return 0.0
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
