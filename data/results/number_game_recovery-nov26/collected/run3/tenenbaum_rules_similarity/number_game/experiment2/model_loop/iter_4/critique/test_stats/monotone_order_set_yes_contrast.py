# name: monotone_order_set_yes_contrast
# description: Among sets of >=3 examples and non-example targets, yes rate for sets shown in strictly ascending or descending order minus sets shown in mixed order; observed > null means sorted presentation makes people generalise more broadly than the order-invariant model predicts from content alone, observed < null means sorted sets are read as narrower sequences.
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    mono = {}
    for s in pd.unique(sets):
        e = [int(x) for x in s.split(",")]
        if len(e) < 3:
            mono[s] = -1
        else:
            inc = all(a < b for a, b in zip(e, e[1:]))
            dec = all(a > b for a, b in zip(e, e[1:]))
            mono[s] = 1 if (inc or dec) else 0
    code = pd.Series(sets).map(mono).values
    isex = np.array([str(x) in s.split(",") for x, s in zip(t, sets)])
    a = r[(code == 1) & ~isex]
    b = r[(code == 0) & ~isex]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
