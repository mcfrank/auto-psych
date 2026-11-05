# name: middle_example_digit_lift
# description: In 3-4 example sets without a shared final digit, yes-rate for targets sharing only a MIDDLE example's final digit (not first/last) minus yes-rate for targets sharing no example's final digit; observed > null means the model under-credits middle examples' digits (it only lets first/last digits linger).
def test_statistic(df):
    P = df["number_set"].astype(str).str.split(",", expand=True)
    X = np.column_stack([pd.to_numeric(P[c], errors="coerce").to_numpy(float) for c in P.columns])
    n = (~np.isnan(X)).sum(1)
    t = df["target"].astype(int).to_numpy(); r = df["rating"].to_numpy(float)
    D = np.where(np.isnan(X), -1, X % 10)
    idx = np.arange(len(t))
    first = D[:, 0]; last = D[idx, np.maximum(n - 1, 0)]
    td = t % 10
    match = (D == td[:, None])
    allsame = np.array([True] * len(t))
    for c in range(1, D.shape[1]):
        allsame &= (D[:, c] == first) | (D[:, c] == -1)
    base = (n >= 3) & ~allsame
    cols = np.arange(D.shape[1])[None, :]
    midmask = (cols >= 1) & (cols <= (n - 2)[:, None])
    mid = (match & midmask).any(1) & (td != first) & (td != last)
    none = ~match.any(1)
    a = base & mid; b = base & none
    if a.sum() < 5 or b.sum() < 5:
        return 0.0
    return float(r[a].mean() - r[b].mean())
