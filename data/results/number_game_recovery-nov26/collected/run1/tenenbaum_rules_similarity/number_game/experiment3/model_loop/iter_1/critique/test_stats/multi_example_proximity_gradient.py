# name: multi_example_proximity_gradient
# description: On sets of 2+ examples, yes rate for non-example targets within 3 of the nearest example minus yes rate for targets 15+ away from every example; observed above null means people generalise by numeric proximity more than the model predicts, below means less (people less similarity-driven than the model's ranges).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = pd.unique(sets)
    codes = pd.Categorical(sets, categories=uniq).codes
    table = np.full((len(uniq), 4), np.nan)
    for i, s in enumerate(uniq):
        v = [int(x) for x in s.split(",")][:4]
        table[i, :len(v)] = v
    ex = table[codes]
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    a = (n > 1) & (d >= 1) & (d <= 3)
    b = (n > 1) & (d >= 15)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
