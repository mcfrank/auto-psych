# name: outlier_sets_in_range_nonrule
# description: For 4-example sets where exactly 3 examples share a last digit and the 4th does not (e.g. 92,82,27,42; 71,61,26,81), yes-rate for non-example targets strictly inside [min,max] that lack the shared last digit; observed > null means people abandon the near-fitting rule for a range spanning all examples more than the model's inferred stray tolerance predicts (observed < null: they keep the rule and treat the odd example as a stray more than the model).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    if ex.shape[1] < 4:
        return 0.0
    n = (~np.isnan(ex)).sum(axis=1)
    dig = np.where(np.isnan(ex), -1, np.nan_to_num(ex) % 10)
    counts = np.stack([(dig == k).sum(axis=1) for k in range(10)], axis=1)
    has3 = (counts == 3).any(axis=1) & (n == 4)
    maj = counts.argmax(axis=1)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    is_ex = (ex == t[:, None]).any(axis=1)
    mask = has3 & (t % 10 != maj) & ~is_ex & (t > lo) & (t < hi)
    return float(r[mask].mean()) if mask.any() else 0.0
