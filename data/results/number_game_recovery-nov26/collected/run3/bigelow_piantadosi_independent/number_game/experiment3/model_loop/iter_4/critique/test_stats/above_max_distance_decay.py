# name: above_max_distance_decay
# description: On 2+ example sets, yes rate for non-example targets 1-10 above the largest example minus targets more than 20 above it; observed above null means the forward lean is local continuation that decays with distance (the model's lean is flat), below null means far upward targets are accepted more than predicted.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    n = (~np.isnan(ex)).sum(1)
    mx = np.nanmax(ex, 1); mn = np.nanmin(ex, 1)
    isex = (ex == t[:, None]).any(1)
    m = (n >= 2) & ~isex
    gap = t - mx
    a = r[m & (gap >= 1) & (gap <= 10)]; b = r[m & (gap > 20)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
