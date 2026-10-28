# name: clustered_set_near_breadth
# description: For sets of 2+ examples whose span is at most 10, yes rate for non-example targets 1-10 outside the span; this measures how far people generalize from tight clusters, so observed above null_mean means people generalize more broadly from clustered examples than the model predicts, below means more narrowly.
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
    out_d = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    base = (n_ex >= 2) & ((hi - lo) <= 10) & (out_d >= 1) & (out_d <= 10)
    if base.sum() == 0:
        return 0.0
    return float(y[base].mean())
