# name: near_fit_rule_robustness
# description: In sets of >=3 examples that no small rule fits fully but where some small rule fits all but one example, yes rate for non-example targets in such a near-fitting rule; observed above null means people tolerate a stray example and keep the rule more than the model's fixed noise allows, below means they abandon it more readily.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    pad = np.isnan(E)
    n = (~pad).sum(axis=1)
    lo = np.nanmin(E, axis=1)
    hi = np.nanmax(E, axis=1)
    in_set = (E == t[:, None]).any(axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    def rate(mask):
        return float(y[mask].mean()) if mask.any() else 0.0
    N = np.arange(1, 101)
    primes = [k for k in N if k > 1 and all(k % a for a in range(2, int(k ** 0.5) + 1))]
    rules = [[k * k for k in range(1, 11)], [k ** 3 for k in range(1, 5)], primes]
    rules += [list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    rules += [[m ** k for k in range(1, 8) if m ** k <= 100] for m in range(2, 11)]
    rules += [[k for k in N if k % 10 == dd] for dd in range(10)]
    R = np.array([np.isin(N, r) for r in rules])
    R = R[R.sum(axis=1) <= 25]  # small mathematical rules only (no parity / big sets)
    Ei = np.where(pad, 1, E).astype(int) - 1
    fit = (R[:, Ei] & ~pad[None]).sum(axis=2).T  # rows x rules: examples fitting each rule
    t_in = R[:, t - 1].T  # rows x rules: target in rule
    full = fit == n[:, None]
    near = (fit == (n - 1)[:, None]) & t_in
    m = (n >= 3) & ~full.any(axis=1) & ~in_set & near.any(axis=1)
    return rate(m)
