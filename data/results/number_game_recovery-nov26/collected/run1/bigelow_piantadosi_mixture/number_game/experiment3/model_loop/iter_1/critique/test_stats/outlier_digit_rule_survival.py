# name: outlier_digit_rule_survival
# description: Among sets of >=3 examples where all but exactly one example share a last digit (e.g. 77,17,8), yes rate for non-member targets with that majority last digit minus yes rate for other non-member targets; observed > null means people keep the majority digit rule despite the outlier more than the model's misfit tolerance predicts, observed < null less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4))
    E = ex.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)  # (rows, 4), NaN padded
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    in_set = np.any(E == t[:, None], axis=1)
    ld = np.mod(E, 10)
    best = np.full(len(t), -1.0)
    for d in range(10):
        cnt = np.sum(ld == d, axis=1)
        best = np.where((n >= 3) & (cnt == n - 1), d, best)
    ok = best >= 0
    match = (t % 10) == best
    a = ok & ~in_set & match; b = ok & ~in_set & ~match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
