# name: two_digit_overlap_contrast
# description: Among non-example two-digit targets 10-99 with two distinct digits, yes rate when both of the target's digits occur among the examples' digits minus yes rate when exactly one does; observed above null means people reward multiple digit overlap (e.g. digit reversals like 38/83) more than the model's sum of single contains-digit rules, below means less.
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
    two = (t >= 10) & (t <= 99) & ((t // 10) != (t % 10)) & ~is_ex
    tens_in = ((union >> (t // 10)) & 1) == 1
    units_in = ((union >> (t % 10)) & 1) == 1
    return rate(two & tens_in & units_in) - rate(two & (tens_in ^ units_in))
