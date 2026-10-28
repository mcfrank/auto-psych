# name: recency_last_vs_first
# description: Among multi-example sets, the yes rate for targets numerically closer to the last-shown example than to the first, minus the yes rate for targets closer to the first; the model predicts a positive recency gap, so observed < model means people weight examples less by recency (or show primacy) than the model assumes.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    f, l = {}, {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        f[s], l[s] = ex[0], ex[-1]
    F = sets.map(f).to_numpy(); Lv = sets.map(l).to_numpy()
    k = (sets.str.count(",") + 1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    dl, dfi = np.abs(t - Lv), np.abs(t - F)
    m = k >= 2
    a, b = m & (dl < dfi), m & (dfi < dl)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
