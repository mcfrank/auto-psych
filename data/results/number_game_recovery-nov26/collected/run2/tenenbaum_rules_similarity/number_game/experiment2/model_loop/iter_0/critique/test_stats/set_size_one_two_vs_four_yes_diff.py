# name: set_size_one_two_vs_four_yes_diff
# description: Yes rate on 1-2 example sets minus yes rate on 4-example sets (non-example targets); observed above the null means people generalize more broadly from few examples / tighten less with more examples than the model's size principle implies (below = they tighten more).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).values
    notin = ~(ex == t[:, None]).any(axis=1)
    r = df["rating"].values.astype(float)
    a = (n <= 2) & notin
    b = (n == 4) & notin
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
