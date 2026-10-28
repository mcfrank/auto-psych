# name: largest_gap_interior_yes
# description: In sets of >=3 examples, yes-rate for non-example targets inside the span that fall in the largest gap between adjacent sorted examples minus those in the other gaps; observed below null means people treat widely separated example clusters as separate groups (not one range) more than the model, above null means the opposite.
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    dist = np.maximum(mn - t, t - mx)
    S = np.sort(X, axis=1)  # NaN sorted last
    lo = S[:, :-1]
    hi = S[:, 1:]
    gaps = np.where(np.isnan(hi) | np.isnan(lo), -1.0, hi - lo)
    gi = np.argmax(gaps, axis=1)
    rows = np.arange(len(t))
    glo = lo[rows, gi]
    ghi = hi[rows, gi]
    inside = (t > mn) & (t < mx) & ~is_ex & (n >= 3)
    in_big = (t > glo) & (t < ghi)
    a = inside & in_big
    b = inside & ~in_big
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
