# name: sample_size_breadth_contrast
# description: Among targets outside the examples' span by 5-40, yes-rate for four-example sets minus yes-rate for one-example sets; observed < null means people narrow generalization with more examples (size principle) more than the model does, observed > null the reverse.
def test_statistic(df):
    k = df["number_set"].astype(str).str.count(",") + 1
    ex = df["number_set"].astype(str).str.split(",")
    mn = ex.map(lambda e: min(int(x) for x in e))
    mx = ex.map(lambda e: max(int(x) for x in e))
    t = df["target"].astype(int)
    gap = np.maximum(np.maximum(mn - t, t - mx), 0)
    sel = (gap >= 5) & (gap <= 40)
    r = df["rating"].astype(float)
    a = r[sel & (k == 4)]
    b = r[sel & (k == 1)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
