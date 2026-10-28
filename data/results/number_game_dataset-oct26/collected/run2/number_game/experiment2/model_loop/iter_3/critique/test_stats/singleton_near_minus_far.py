# name: singleton_near_minus_far
# description: For single-example sets, non-identical targets: yes-rate within 10 of the example minus more than 30 away; observed above the null means a lone example yields sharper size generalization than the model's (span-shared) magnitude tuning allows, below means people generalize from a lone example more broadly than modelled.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    single = ~ns.str.contains(",").values
    x = pd.to_numeric(ns.where(single, "0")).values
    t = df["target"].astype(int).values
    d = np.abs(t - x)
    y = df["rating"].astype(float).values
    a = single & (d >= 1) & (d <= 10)
    b = single & (d > 30)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
