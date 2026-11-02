# name: n_examples_slope_multi
# description: OLS slope of yes on number of examples (2-4 example sets only, targets not in the set); observed above null means people's generalization narrows less (or widens) with more examples than the model's size principle predicts, below means it narrows more.
def test_statistic(df):
    s = df["number_set"].astype(str)
    n = (s.str.count(",") + 1).to_numpy().astype(float)
    ex = s.str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()[:, None]
    in_set = (ex == t).any(axis=1)
    m = (n >= 2) & ~in_set
    x = n[m]
    y = df["rating"].to_numpy().astype(float)[m]
    if len(x) < 2:
        return 0.0
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    if denom == 0:
        return 0.0
    return float((xc * (y - y.mean())).sum() / denom)
