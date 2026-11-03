# name: set_size_yes_slope
# description: OLS slope of yes-rate on number of examples (2-4) for non-example targets; observed more negative than the null means people tighten generalization with more examples faster than the model's size principle/noise rate predicts, less negative means slower.
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    is_ex = ex.eq(t, axis=0).any(axis=1)
    m = (~is_ex) & (n >= 2)
    x = n[m].astype(float).values
    y = df.loc[m, "rating"].astype(float).values
    if len(x) < 3 or np.var(x) == 0:
        return 0.0
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
