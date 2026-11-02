# name: single_example_parity_match
# description: For single-example sets, among non-example targets more than 3 from the example and not sharing its final digit, yes-rate for same-parity targets minus opposite-parity targets; observed above the null means people generalize a lone example's parity (odd/even rule) more than the model, below means the model over-weights parity rules for one example.
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
    same_par = np.mod(np.nan_to_num(E[:, 0]), 2) == np.mod(t, 2)
    m = (~is_ex) & (n == 1) & (~share_fd) & (dist > 3)
    return rate(m & same_par) - rate(m & (~same_par))
