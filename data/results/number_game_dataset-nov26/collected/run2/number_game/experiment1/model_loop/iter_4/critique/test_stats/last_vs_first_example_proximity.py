# name: last_vs_first_example_proximity
# description: In multi-example sets, yes-rate for targets numerically closer to the last-shown example than the first minus targets closer to the first; the model ignores example order, so observed > null means a recency effect (the latest example anchors generalization), observed < null a primacy effect.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    f, l = {}, {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        f[s], l[s] = ex[0], ex[-1]
    t = df["target"].astype(int).to_numpy()
    df_ = np.abs(t - sets.map(f).to_numpy())
    dl = np.abs(t - sets.map(l).to_numpy())
    y = df["rating"].astype(float).to_numpy()
    multi = (sets.str.count(",") >= 1).to_numpy()
    a = multi & (dl < df_)
    b = multi & (df_ < dl)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
