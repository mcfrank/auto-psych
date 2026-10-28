# name: shared_last_digit_rule_contrast_multi
# description: For sets of 2+ examples that all end in the same digit, yes-rate on non-example targets with that last digit minus yes-rate on targets without it; observed > model means people apply the "ends in d" rule more sharply after multiple examples than the model predicts, observed < model means they apply it more weakly (more similarity/range-based).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    ld = np.mod(ex, 10)
    same = np.nanmax(ld, axis=1) == np.nanmin(ld, axis=1)
    d0 = np.mod(ex[:, 0], 10)
    is_ex = np.any(ex == t[:, None], axis=1)
    base = (n >= 2) & same & ~is_ex
    a = base & (t % 10 == d0); b = base & (t % 10 != d0)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
