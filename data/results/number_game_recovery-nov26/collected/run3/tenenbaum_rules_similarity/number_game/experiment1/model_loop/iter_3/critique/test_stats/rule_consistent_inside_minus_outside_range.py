# name: rule_consistent_inside_minus_outside_range
# description: Among multi-example sets whose examples all share a last digit or are all multiples of some m in 3..12, for non-example targets that satisfy that same rule, yes-rate when the target lies inside the examples' [min,max] minus when it lies outside; observed above model means people restrict an arithmetic rule to the examples' magnitude range (rule-within-range) more than the model's rule hypotheses (which ignore range) allow, below means less.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True)
    ex = parts.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n_ex = np.sum(~np.isnan(ex), axis=1)
    exz = np.nan_to_num(ex, nan=0.0)
    present = ~np.isnan(ex)
    t = df["target"].astype(int).to_numpy()
    consistent = np.zeros(len(df), dtype=bool)
    ld = exz % 10
    same_ld = np.all(~present | (ld == ld[:, [0]]), axis=1)
    consistent |= same_ld & (t % 10 == ld[:, 0])
    for m in range(3, 13):
        allm = np.all(~present | (exz % m == 0), axis=1)
        consistent |= allm & (t % m == 0)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    base = (n_ex >= 2) & consistent & ~is_ex
    inside = base & (t > lo) & (t < hi)
    outside = base & ((t < lo) | (t > hi))
    r = df["rating"].to_numpy()
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    return float(r[inside].mean() - r[outside].mean())
