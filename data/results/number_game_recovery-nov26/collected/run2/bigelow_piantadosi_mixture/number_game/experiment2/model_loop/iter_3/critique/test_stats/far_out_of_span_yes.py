# name: far_out_of_span_yes
# description: Yes rate for targets more than 15 beyond the examples' span (below min or above max) on sets of 2+ examples; observed above null means people keep extending to the number-line ends more than the open-ended-range discount allows (observed below null: the model still over-extends far).
def _m(y, m):
    return float(y[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = np.sum(~np.isnan(X), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(X, axis=1)
    mx = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    gap = np.maximum(mn - t, t - mx)
    return _m(y, (n >= 2) & (gap > 15))
