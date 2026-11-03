# name: multiple_of_example_targets
# description: In 1-2 example sets, among non-example targets more than 5 from every example and sharing no final digit with any example, yes-rate for targets that are a larger multiple of some example minus other targets; observed above the null means people generalize by "multiples of the example" more than the model's fixed rule list (multiples of 3-12 only) allows.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    fd = np.mod(E, 10)
    share_fd = np.any(fd == np.mod(t, 10)[:, None], axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    mul = np.any((np.mod(t[:, None], E) == 0) & (t[:, None] > E), axis=1)
    m = (~is_ex) & (n <= 2) & (dist > 5) & (~share_fd)
    return rate(m & mul) - rate(m & ~mul)
