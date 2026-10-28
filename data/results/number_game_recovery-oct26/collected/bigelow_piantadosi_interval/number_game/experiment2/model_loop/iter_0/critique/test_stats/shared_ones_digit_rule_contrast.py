# name: shared_ones_digit_rule_contrast
# description: Among sets of >= 2 examples that all share one ones digit, yes rate for non-example targets with that ones digit minus yes rate for targets without it; observed > null means people pick up the "ends in d" rule more strongly than the model (observed < null means the model over-commits to the digit rule).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    tg = df["target"].astype(int).values.astype(float)
    k = (~np.isnan(ex)).sum(axis=1)
    d = np.mod(ex, 10)
    d0 = d[:, 0]
    same = np.all(np.isnan(d) | (d == d0[:, None]), axis=1) & (k >= 2)
    isex = (ex == tg[:, None]).any(axis=1)
    base = same & ~isex
    match = base & (np.mod(tg, 10) == d0)
    nomatch = base & (np.mod(tg, 10) != d0)
    r = df["rating"].values.astype(float)
    if match.sum() == 0 or nomatch.sum() == 0:
        return 0.0
    return float(r[match].mean() - r[nomatch].mean())
