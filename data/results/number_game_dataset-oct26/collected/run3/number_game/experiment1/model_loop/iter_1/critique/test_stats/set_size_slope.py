# name: set_size_slope
# description: OLS slope of yes (0/1) on the number of examples (1-4), over targets not among the examples; observed more negative than null means people tighten generalization with more examples faster than the single evidence-strength model predicts (less negative = slower).
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric)
    n = s.notna().sum(axis=1).to_numpy(float)
    t = df["target"].astype(int)
    ok = ~s.eq(t, axis=0).any(axis=1).to_numpy()
    x, y = n[ok], df["rating"].to_numpy(float)[ok]
    if len(x) < 3 or np.var(x) == 0:
        return 0.0
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
