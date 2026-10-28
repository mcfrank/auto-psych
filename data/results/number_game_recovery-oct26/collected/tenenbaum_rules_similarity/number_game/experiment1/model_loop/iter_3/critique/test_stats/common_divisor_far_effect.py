# name: common_divisor_far_effect
# description: For sets of 2+ examples whose greatest common divisor is at least 3, among targets more than 10 from every example, yes rate for multiples of that divisor minus non-multiples; observed above null_mean means people generalize by shared-divisor rules more than the model predicts, below means less.
def test_statistic(df):
    ex = pd.DataFrame(df["number_set"].astype(str).str.split(",").tolist(), index=df.index).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n_ex = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    def diff(a, b):
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return float(y[a].mean() - y[b].mean())
    e = np.nan_to_num(ex, nan=0.0).astype(np.int64)
    g = np.zeros(len(t), dtype=np.int64)
    for j in range(e.shape[1]):
        g = np.gcd(g, e[:, j])
    base = (n_ex >= 2) & (g >= 3) & (dist > 10)
    mult = np.mod(t.astype(np.int64), np.maximum(g, 1)) == 0
    return diff(base & mult, base & ~mult)
