# name: yes_rate_slope_on_n_examples
# description: OLS slope of the yes response on the number of examples (1-4); observed more negative than null means humans tighten generalisation with more examples faster than the model, less negative means slower.
def _ex(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values.astype(float)
    r = df["rating"].astype(float).values
    return ex, t, r

def test_statistic(df):
    ex, t, r = _ex(df)
    n = np.sum(~np.isnan(ex), axis=1).astype(float)
    if n.std() == 0: return 0.0
    return float(np.cov(n, r, bias=True)[0, 1] / n.var())
