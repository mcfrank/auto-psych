# name: warmup_yes_difference
# description: Yes rate on early trials (trial_index < 50) minus yes rate on all later trials; observed above null means people say yes more at the start of the session (a warm-up / calibration effect the stationary model cannot produce), below means they start more conservative.
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
    ti = df["trial_index"].astype(int).to_numpy()
    return rate(ti < 50) - rate(ti >= 50)
