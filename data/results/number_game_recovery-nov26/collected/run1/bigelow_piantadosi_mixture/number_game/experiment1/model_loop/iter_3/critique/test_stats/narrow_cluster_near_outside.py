# name: narrow_cluster_near_outside
# description: For multi-example sets spanning <= 30 (tight clusters), yes-rate for targets 1-20 outside the examples' range; observed > null means people extend tight clusters beyond their edges more than the model's size-principle interval guesses allow, observed < null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = np.isfinite(ex).sum(axis=1)
    t = df["target"].astype(float).to_numpy(); r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    out = np.maximum(lo - t, t - hi)
    m = (n >= 2) & (hi - lo <= 30) & (out >= 1) & (out <= 20)
    if not m.any():
        return 0.0
    return float(r[m].mean())
