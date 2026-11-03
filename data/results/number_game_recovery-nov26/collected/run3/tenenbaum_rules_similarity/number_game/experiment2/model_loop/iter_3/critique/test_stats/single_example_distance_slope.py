# name: single_example_distance_slope
# description: For single-example sets, OLS slope of yes (per 10 units) on |target - example| over targets != example; more negative observed than null means people's generalisation from one example falls off with numeric distance more steeply than the model, less negative means flatter (more rule-like) generalisation.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    single = ~sets.str.contains(",")
    x = pd.to_numeric(sets[single], errors="coerce").values
    t = df["target"].astype(float).values[single.values]
    r = df["rating"].astype(float).values[single.values]
    keep = np.isfinite(x) & (t != x)
    d = np.abs(t[keep] - x[keep]) / 10.0
    y = r[keep]
    if len(d) < 3 or np.var(d) == 0:
        return 0.0
    return float(np.cov(d, y, bias=True)[0, 1] / np.var(d))
