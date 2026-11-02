# name: repdigit_target_generalization
# description: In sets containing a repdigit example (11,22,...,99), yes-rate for non-example repdigit targets sharing no final digit with any example minus other non-example targets sharing no final digit and more than 3 from every example; observed above the null means people use a 'same two digits' rule the model lacks, below means they avoid those targets more than predicted.
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
    rep_ex = np.any((E >= 11) & (np.mod(E, 11) == 0), axis=1)
    rep_t = (t >= 11) & (np.mod(t, 11) == 0)
    m = (~is_ex) & rep_ex & (~share_fd)
    return rate(m & rep_t) - rate(m & (~rep_t) & (dist > 3))
