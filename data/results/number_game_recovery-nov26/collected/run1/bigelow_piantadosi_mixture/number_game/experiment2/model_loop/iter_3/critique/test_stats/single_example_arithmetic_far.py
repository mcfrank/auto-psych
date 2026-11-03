# name: single_example_arithmetic_far
# description: Single-example sets: yes-rate for targets >10 away, not sharing the example's last digit, that share a small arithmetic rule with it (common multiple-of-m for m in 3..12, square, cube, or power of 2-10); observed above null_mean means the model under-produces arithmetic-rule generalisation from one example, below means it over-produces it.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    isex = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    dmin = np.nanmin(np.abs(E - t[:, None]), axis=1)
    ld = np.mod(E, 10)
    x = E[:, 0]
    hit = np.zeros(len(t), dtype=bool)
    for m_ in range(3, 13):
        if m_ == 10:
            continue
        hit |= (np.mod(x, m_) == 0) & (np.mod(t, m_) == 0)
    sq = np.array([k * k for k in range(1, 11)], float)
    cu = np.array([1, 8, 27, 64], float)
    hit |= np.isin(x, sq) & np.isin(t, sq)
    hit |= np.isin(x, cu) & np.isin(t, cu)
    for b in range(2, 11):
        p = np.array([b ** k for k in range(1, 8) if b ** k <= 100], float)
        hit |= np.isin(x, p) & np.isin(t, p)
    m = (n == 1) & hit & (np.abs(t - x) > 10) & (np.mod(t, 10) != np.mod(x, 10))
    return float(r[m].mean()) if m.sum() > 0 else 0.0
