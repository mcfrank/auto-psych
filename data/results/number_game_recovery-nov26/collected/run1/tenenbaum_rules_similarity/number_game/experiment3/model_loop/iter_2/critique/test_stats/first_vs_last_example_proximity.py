# name: first_vs_last_example_proximity
# description: Among multi-example sets, yes rate for targets strictly closer to the first-shown example than to the last-shown example minus yes rate for targets strictly closer to the last; observed above null means people weight the first example (primacy) more than the model, below means they weight the last example (recency) more than the model.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = pd.unique(sets)
    codes = pd.Categorical(sets, categories=uniq).codes
    fl = np.zeros((len(uniq), 2)); multi = np.zeros(len(uniq), bool)
    for i, s in enumerate(uniq):
        v = [int(x) for x in s.split(",")]
        fl[i] = [v[0], v[-1]]; multi[i] = len(v) > 1
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    d1 = np.abs(t - fl[codes, 0]); d2 = np.abs(t - fl[codes, 1])
    m = multi[codes]
    a, b = m & (d1 < d2), m & (d2 < d1)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
