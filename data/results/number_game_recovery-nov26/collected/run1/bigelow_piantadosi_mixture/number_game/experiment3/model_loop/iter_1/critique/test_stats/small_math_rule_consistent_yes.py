# name: small_math_rule_consistent_yes
# description: Among sets with >=2 examples, yes rate for non-member targets that fall in a small arithmetic rule (powers, squares, cubes, multiples of 3-12, size<=34) containing every example; observed > null means people endorse rule-consistent targets more than the model predicts (model under-weights arithmetic rules), observed < null the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4))
    E = ex.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)  # (rows, 4), NaN padded
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    in_set = np.any(E == t[:, None], axis=1)
    N = np.arange(1, 101)
    rules = [[k * k for k in range(1, 11)], [k ** 3 for k in range(1, 5)]]
    for b in range(2, 11):
        rules.append([b ** k for k in range(1, 8) if b ** k <= 100])
    for m in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        rules.append(list(range(m, 101, m)))
    M = np.zeros((101, len(rules)))
    for j, r in enumerate(rules):
        M[r, j] = 1.0
    M[0, :] = 1.0  # padding index counts as member
    idx = np.nan_to_num(E, nan=0).astype(int)
    all_in = np.prod(M[idx], axis=1)  # (rows, rules)
    hit = (all_in * M[t]).sum(axis=1) > 0
    mask = (n >= 2) & ~in_set & hit
    if mask.sum() == 0:
        return 0.0
    return float(y[mask].mean())
