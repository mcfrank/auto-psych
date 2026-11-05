# name: digit_run_far_match_gradient
# description: In sets of 3+ examples all sharing one final digit, yes-rate for same-final-digit targets >=20 from the nearest example minus same-digit targets <=10 from it (non-examples only); observed > null (less negative) means people apply the digit rule regardless of distance, whereas the model's coincidence sharpening also sharpens magnitude and so over-penalizes far rule-matching targets.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t1 = df["target"].astype(int).to_numpy().astype(float)
    t = t1[:, None]
    pres = ~np.isnan(ex)
    n = pres.sum(1)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    r = df["rating"].to_numpy(dtype=float)
    d = np.abs(t - e).min(1)
    digmatch = ((np.mod(t, 10) == np.mod(e1, 10)) & pres).any(1)
    def diff(a, b):
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return float(r[a].mean() - r[b].mean())
    u0 = np.mod(e1[:, 0], 10)
    run = (~pres | (np.mod(e1, 10) == u0[:, None])).all(1) & (n >= 3)
    same = np.mod(t1, 10) == u0
    isex = ((t == e) & pres).any(1)
    base = run & same & ~isex
    return diff(base & (d >= 20), base & (d <= 10))
