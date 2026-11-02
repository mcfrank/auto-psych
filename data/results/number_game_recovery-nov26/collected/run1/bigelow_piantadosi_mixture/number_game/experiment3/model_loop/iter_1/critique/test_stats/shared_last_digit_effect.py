# name: shared_last_digit_effect
# description: Among sets of >=2 examples that all share one last digit, yes rate for non-member targets with that last digit minus yes rate for other targets; observed > null means people generalize a shared final digit more strongly than the model predicts (model over-discounts digit rules), observed < null the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4))
    E = ex.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)  # (rows, 4), NaN padded
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    in_set = np.any(E == t[:, None], axis=1)
    ld = np.mod(E, 10)
    first = ld[:, 0]
    same = np.all(np.isnan(ld) | (ld == first[:, None]), axis=1) & (n >= 2)
    match = (t % 10) == first
    a = same & ~in_set & match; b = same & ~in_set & ~match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
