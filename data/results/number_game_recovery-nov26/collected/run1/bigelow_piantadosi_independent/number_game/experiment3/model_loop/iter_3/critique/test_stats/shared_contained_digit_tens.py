# name: shared_contained_digit_tens
# description: In 2-4 example sets where every example contains a common digit d, among non-example targets more than 5 from every example and sharing no final digit with any example, yes-rate for targets whose tens digit is such a d minus other such targets; observed above the null means people use 'contains digit d' rules more than the model, below means the model over-weights those rules.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    share_fd = np.any(np.mod(E, 10) == np.mod(t, 10)[:, None], axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    Ei = np.nan_to_num(E, nan=-1).astype(int)
    valid = Ei >= 0
    common = np.ones((len(t), 10), dtype=bool)
    for d in range(10):
        has = (np.mod(Ei, 10) == d) | ((Ei // 10) % 10 == d) & (Ei >= 10) | (Ei == 100) & (d in (0, 1))
        common[:, d] = np.all(has | ~valid, axis=1)
    ti = t.astype(int)
    tens = np.where(ti >= 10, (ti // 10) % 10, -1)
    hit = np.zeros(len(t), dtype=bool)
    for d in range(10):
        hit |= common[:, d] & (tens == d)
    m = (~is_ex) & (n >= 2) & (~share_fd) & (dist > 5) & common.any(axis=1)
    return rate(m & hit) - rate(m & (~hit))
