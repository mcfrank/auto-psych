# name: partial_last_digit_match_effect
# description: In multi-example sets whose examples do NOT all share a last digit, yes rate for targets sharing a last digit with at least one example minus yes rate for other targets; observed above null means people generalize by partial last-digit similarity more than the all-or-none rule model predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    ld = np.mod(ex, 10)
    first = ld[:, :1]
    same_all = np.all((ld == first) | np.isnan(ld), axis=1)
    sel = (n >= 2) & ~same_all & ~np.any(ex == t[:, None], axis=1)
    match = np.any(ld == np.mod(t, 10)[:, None], axis=1)
    a, b = sel & match, sel & ~match
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
