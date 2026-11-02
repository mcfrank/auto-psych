# name: single_example_magnitude_neighbourhood
# description: For one-example sets, among targets sharing no digit with the example, yes rate for targets within 10 of the example minus targets more than 30 away; observed above null means people generalize from a single number by magnitude similarity more than the model's interval hypotheses (outcompeted by digit rules) allow, below means less.
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
    m = (n_ex == 1) & ~is_ex & ((TM & union) == 0)
    dist = np.abs(Ei[:, 0] - t)
    return rate(m & (dist <= 10)) - rate(m & (dist > 30))
