# name: small_vs_large_unrelated_target
# description: Among non-example targets sharing no digit with any example and more than 15 from every example, yes rate for targets <=30 minus targets >=70; observed above null means people favour small numbers a priori (a magnitude bias the model lacks), below means they favour large numbers more than the model predicts.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    present = ~np.isnan(E)
    Ei = np.where(present, E, 0).astype(int)
    def dmask(a):
        m = (1 << (a % 10)) | np.where(a >= 10, 1 << ((a // 10) % 10), 0) | np.where(a >= 100, 2, 0)
        return np.where(a > 0, m, 0)
    EM = dmask(Ei)
    union = np.bitwise_or.reduce(EM, axis=1)
    inter = np.bitwise_and.reduce(np.where(present, EM, 1023), axis=1)
    t = df["target"].astype(int).to_numpy()
    TM = dmask(t)
    r = df["rating"].astype(float).to_numpy()
    n_ex = present.sum(axis=1)
    is_ex = (Ei == t[:, None]).any(axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    dist = np.where(present, np.abs(Ei - t[:, None]), 999).min(axis=1)
    base = ~is_ex & ((TM & union) == 0) & (dist > 15)
    return rate(base & (t <= 30)) - rate(base & (t >= 70))
