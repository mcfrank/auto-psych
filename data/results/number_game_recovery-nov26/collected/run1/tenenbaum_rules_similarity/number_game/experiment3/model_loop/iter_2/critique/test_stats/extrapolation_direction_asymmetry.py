# name: extrapolation_direction_asymmetry
# description: Among multi-example sets, yes rate for targets above the largest example minus yes rate for targets below the smallest example; observed above null means people extend generalisation upward (toward larger numbers) more than the model predicts, below means they extend downward more than the model.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = pd.unique(sets)
    codes = pd.Categorical(sets, categories=uniq).codes
    mm = np.zeros((len(uniq), 2)); multi = np.zeros(len(uniq), bool)
    for i, s in enumerate(uniq):
        v = [int(x) for x in s.split(",")]
        mm[i] = [min(v), max(v)]; multi[i] = len(v) > 1
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    m = multi[codes]
    a, b = m & (t > mm[codes, 1]), m & (t < mm[codes, 0])
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
