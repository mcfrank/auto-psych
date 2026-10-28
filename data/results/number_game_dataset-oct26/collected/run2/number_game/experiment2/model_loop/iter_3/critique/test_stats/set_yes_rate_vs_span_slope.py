# name: set_yes_rate_vs_span_slope
# description: Across 2+-example sets, OLS slope of each set's overall yes-rate on its span (max-min, per 100); observed below the null means the model's span-filling makes widely spread sets too generous relative to tight ones (people broaden less with span), above means people broaden with span even more than modelled.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    g = df["rating"].astype(float).groupby(ns).mean()
    sets = g.index.values
    k = np.array([s.count(",") + 1 for s in sets])
    span = np.array([max(int(v) for v in s.split(",")) - min(int(v) for v in s.split(",")) for s in sets]) / 100.0
    m = k >= 2
    x, yy = span[m], g.values[m]
    if m.sum() < 3 or np.var(x) == 0:
        return 0.0
    return float(np.cov(x, yy, bias=True)[0, 1] / np.var(x))
