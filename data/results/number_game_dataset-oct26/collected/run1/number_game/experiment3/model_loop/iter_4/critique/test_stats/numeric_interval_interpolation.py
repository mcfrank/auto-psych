# name: numeric_interval_interpolation
# description: In multi-example sets with no shared-digit rule and no common factor >=2, among targets sharing no example's final digit, yes-rate for targets strictly inside the examples' numeric range [min,max] minus those outside it; observed > null means people use the fine-grained numeric interval spanned by the examples more than the model's decade-band familiarity captures; observed < null the reverse.
def test_statistic(df):
    X = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy(float)
    valid = ~np.isnan(X)
    n = valid.sum(1)
    Xi = np.nan_to_num(X).astype(int)
    g = np.gcd.reduce(Xi, axis=1)
    d = np.where(valid, Xi % 10, -1)
    dmin = np.where(valid, d, 99).min(1); dmax = np.where(valid, d, -99).max(1)
    same_digit = (valid & (d == (t % 10)[:, None])).any(1)
    lo = np.where(valid, X, np.inf).min(1); hi = np.where(valid, X, -np.inf).max(1)
    base = (n >= 2) & (dmin != dmax) & (g < 2) & ~same_digit
    inside = (t > lo) & (t < hi)
    a = base & inside; b = base & ~inside
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
