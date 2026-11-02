# name: prime_set_prime_vs_odd_composite
# description: For sets of >=2 examples that are all odd primes, yes rate for non-example prime targets minus yes rate for odd composite targets; observed below null means people fail to distinguish primes from other odd numbers as sharply as the model predicts (above: they discriminate more).
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",")
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    mat = np.full((len(df), 4), -1)
    for j in range(4):
        mat[:, j] = pd.to_numeric(s.str[j], errors="coerce").fillna(-1).astype(int).values
    valid = mat > 0
    n = valid.sum(1)
    shown = np.any(valid & (mat == t[:, None]), 1)
    lo = np.where(valid, mat, 999).min(1)
    hi = np.where(valid, mat, -1).max(1)
    P = np.array([k for k in range(2, 101) if all(k % a for a in range(2, int(k ** 0.5) + 1))])
    isp = np.isin(mat, P) & (mat % 2 == 1)
    allp = np.all(~valid | isp, 1) & (n >= 2)
    tp = np.isin(t, P)
    a = allp & tp & ~shown
    b = allp & ~tp & (t % 2 == 1) & (t > 1)
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
