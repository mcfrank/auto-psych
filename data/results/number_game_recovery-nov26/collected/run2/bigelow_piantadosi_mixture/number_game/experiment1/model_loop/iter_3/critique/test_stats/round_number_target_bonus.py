# name: round_number_target_bonus
# description: In sets with no example ending in 0, among non-example targets sharing no digit with any example, yes rate for multiples of 10 minus yes rate for other targets; observed above null means people treat round numbers as a priori salient members more than the model does, below means the model over-produces yes to round numbers.
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
    base = ~is_ex & ((TM & union) == 0) & ~((Ei % 10 == 0) & present).any(axis=1)
    return rate(base & (t % 10 == 0)) - rate(base & (t % 10 != 0))
