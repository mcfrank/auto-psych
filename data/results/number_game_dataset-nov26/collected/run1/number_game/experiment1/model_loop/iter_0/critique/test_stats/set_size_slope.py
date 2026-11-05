# name: set_size_slope
# description: OLS slope of yes on the number of examples in the set (1-4); observed more negative than null_mean means people tighten generalization with more examples more than the model (stronger size-principle), less negative/positive means the model's size principle is too strong.
def test_statistic(df):
    x = (df["number_set"].astype(str).str.count(",") + 1).astype(float).values
    y = df["rating"].astype(float).values
    vx = np.var(x)
    if vx == 0:
        return 0.0
    return float(np.mean((x - x.mean()) * (y - y.mean())) / vx)
