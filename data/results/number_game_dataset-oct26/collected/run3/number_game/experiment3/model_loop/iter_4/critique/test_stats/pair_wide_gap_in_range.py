# name: pair_wide_gap_in_range
# description: For 2-example sets with range >= 20 whose examples share no last digit and no common factor >= 3, yes-rate for non-example targets strictly inside [min,max]; observed > null means people fill the gap between two unrelated distant examples more than the model's interval/rule mixture predicts (observed < null: less).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = (~np.isnan(ex)).sum(axis=1)
    a = np.nan_to_num(ex[:, 0]).astype(int)
    b = np.nan_to_num(ex[:, 1]).astype(int) if ex.shape[1] > 1 else np.zeros_like(a)
    g = np.gcd(a, b)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    ok = (n == 2) & (hi - lo >= 20) & (a % 10 != b % 10) & (g < 3)
    mask = ok & (t > lo) & (t < hi)
    return float(r[mask].mean()) if mask.any() else 0.0
