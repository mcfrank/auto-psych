# name: near_rule_outlier_rule_yes
# description: For 3-4-example sets where exactly all-but-one examples share a last digit or divisibility by 3..10 (and not all do), yes-rate on non-example targets that fit that shared rule and lie outside the range of the rule-fitting examples; observed > model means people keep a near-fitting rule despite an exception more than the model's inferred stray-example tolerance predicts, observed < model means they abandon it more readily.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    n = (~np.isnan(ex)).sum(axis=1)
    tgt = df["target"].astype(float).values
    valid_ex = ~np.isnan(ex)
    exf = np.where(valid_ex, ex, 0.0)
    is_ex = ((exf == tgt[:, None]) & valid_ex).any(axis=1)
    hit = np.zeros(len(df), dtype=bool)
    tests = [("mod10", d) for d in range(10)] + [("div", k) for k in range(3, 11)]
    for kind, v in tests:
        if kind == "mod10":
            fit = (np.mod(exf, 10) == v) & valid_ex
            tfit = np.mod(tgt, 10) == v
        else:
            fit = (np.mod(exf, v) == 0) & valid_ex
            tfit = np.mod(tgt, v) == 0
        cnt = fit.sum(axis=1)
        cond = (n >= 3) & (cnt == n - 1)
        lo = np.where(fit, exf, np.inf).min(axis=1)
        hi = np.where(fit, exf, -np.inf).max(axis=1)
        outside = (tgt < lo) | (tgt > hi)
        hit |= cond & tfit & outside & (~is_ex)
    if hit.sum() == 0:
        return 0.0
    return float(df["rating"].values[hit].mean())
