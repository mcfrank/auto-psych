# name: same_digit_distance_gradient
# description: In 2-4 example sets, among non-example targets sharing a final digit with some example, yes-rate for those within 10 of the nearest example minus those more than 30 from every example; the model's final-digit rules are flat in magnitude, so observed above the null means people's digit-rule generalization decays with distance more than the model predicts (below means less).
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
    m = (~is_ex) & (n >= 2) & share_fd
    return rate(m & (dist <= 10)) - rate(m & (dist > 30))
