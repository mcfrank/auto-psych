# name: far_target_magnitude_asymmetry
# description: In all sets, among non-example targets more than 15 from every example and sharing no final digit with any example, yes-rate for small targets (<=30) minus large targets (>=70); observed above the null means people favour small numbers as far generalizations more than the model's compressed number line predicts, below means the model over-produces the small-number bias.
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
    m = (~is_ex) & (~share_fd) & (dist > 15)
    return rate(m & (t <= 30)) - rate(m & (t >= 70))
