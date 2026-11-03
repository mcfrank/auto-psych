# name: partial_last_digit_match
# description: For sets whose examples do NOT all share a last digit, yes rate among non-example targets sharing a last digit with at least one example minus those sharing none; the model only rewards last-digit matches via full 'ends in d' rules, so observed above null means people use feature-level digit similarity to individual examples more than the model predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)  # (rows, <=4), NaN pad
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    dist = np.abs(ex - t[:, None])
    nearest = np.nanmin(dist, axis=1)
    nonmember = nearest > 0
    ld = np.mod(ex, 10)
    all_same = np.nanmax(ld, axis=1) == np.nanmin(ld, axis=1)
    share = np.any(ld == np.mod(t, 10)[:, None], axis=1)
    m = nonmember & ~all_same
    a, b = r[m & share], r[m & ~share]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
