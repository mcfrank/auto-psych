# name: setsize_yes_slope
# description: OLS slope of yes-rate on number of examples (1-4) across all trials; observed more negative than null means people narrow with more examples faster than the model predicts (and vice versa).
def test_statistic(df):
    n = (df["number_set"].astype(str).str.count(",") + 1).to_numpy(float)
    r = df["rating"].to_numpy(float)
    if np.std(n) == 0:
        return 0.0
    return float(np.cov(n, r)[0, 1] / np.var(n, ddof=1))
