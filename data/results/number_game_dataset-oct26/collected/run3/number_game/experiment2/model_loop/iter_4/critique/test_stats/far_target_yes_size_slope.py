# name: far_target_yes_size_slope
# description: Slope of the yes-rate for targets lying >15 outside the example range against set size (1-4); observed less negative than null means the model shrinks far-target generalisation with more examples too strongly.
def test_statistic(df):
    s = df["number_set"].astype(str)
    sp = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    lo = sp.min(axis=1).values
    hi = sp.max(axis=1).values
    n = sp.notna().sum(axis=1).values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    far = (t < lo - 15) | (t > hi + 15)
    xs, ys = [], []
    for k in (1, 2, 3, 4):
        mk = far & (n == k)
        if mk.sum() > 0:
            xs.append(k); ys.append(r[mk].mean())
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(np.array(xs, float), np.array(ys), 1)[0])
