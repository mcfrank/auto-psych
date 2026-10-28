# name: multi_far_target_yes_floor
# description: Yes-rate on targets more than 25 from every example in sets with 2+ examples; observed above null means people say yes to remote targets (a response floor / loose rule generalization) more than the Bayesian posterior predicts, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    dist = np.abs(ex - t[:, None])
    dmin = np.nanmin(dist, axis=1)
    m = (n >= 2) & (dmin > 25)
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
