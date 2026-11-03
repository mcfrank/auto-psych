# name: first_vs_last_example_digit_share
# description: In multi-example sets, among non-example targets, yes rate for targets sharing a digit with the last-shown example but not the first minus yes rate for targets sharing a digit with the first-shown example but not the last; the model is order-invariant, so observed above null means recency in digit-based generalization, below means primacy.
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
    first = EM[:, 0]
    last_idx = n_ex - 1
    last = EM[np.arange(len(EM)), last_idx]
    m = (n_ex >= 2) & ~is_ex
    sl = (TM & last) > 0
    sf = (TM & first) > 0
    return rate(m & sl & ~sf) - rate(m & sf & ~sl)
