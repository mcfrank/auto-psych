# name: set_range_yes_correlation
# description: Pearson correlation, across multi-example sets with >=10 trials, between the examples' numeric range (max-min) and the set's yes rate on non-example targets; observed above null means people broaden generalization with set spread more than the model (spread matters more), below means less.
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
    rng = np.where(present, Ei, -1).max(axis=1) - np.where(present, Ei, 999).min(axis=1)
    m = (n_ex >= 2) & ~is_ex
    g = pd.DataFrame({"s": df["number_set"].to_numpy()[m], "rng": rng[m], "r": r[m]}).groupby("s").agg(rng=("rng", "first"), y=("r", "mean"), n=("r", "size"))
    g = g[g["n"] >= 10]
    if len(g) < 3 or g["y"].std() == 0 or g["rng"].std() == 0:
        return 0.0
    return float(np.corrcoef(g["rng"], g["y"])[0, 1])
