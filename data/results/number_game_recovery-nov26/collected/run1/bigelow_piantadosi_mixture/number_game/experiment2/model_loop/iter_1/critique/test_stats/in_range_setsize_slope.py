# name: in_range_setsize_slope
# description: OLS slope, across number of examples (2-4), of the yes rate for non-example targets lying between the smallest and largest example; observed below null means people become less willing to fill in the range as examples accumulate, whereas the model predicts more (or flatter) range filling.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = s.notna().sum(axis=1).to_numpy().astype(float)
    t = df["target"].astype(float).to_numpy()
    mn = s.min(axis=1).to_numpy(); mx = s.max(axis=1).to_numpy()
    is_ex = s.eq(pd.Series(t, index=s.index), axis=0).any(axis=1).to_numpy()
    m = (n >= 2) & (t > mn) & (t < mx) & ~is_ex
    if m.sum() < 3:
        return 0.0
    x = n[m]; y = df["rating"].astype(float).to_numpy()[m]
    xc = x - x.mean(); den = (xc ** 2).sum()
    if den <= 0:
        return 0.0
    return float((xc * (y - y.mean())).sum() / den)
