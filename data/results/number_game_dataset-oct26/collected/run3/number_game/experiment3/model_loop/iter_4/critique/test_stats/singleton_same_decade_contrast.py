# name: singleton_same_decade_contrast
# description: For single-example sets, among targets within 9 of the example (excluding the example), yes-rate for targets in the same decade (same tens digit, 100 counted with 90s) minus yes-rate for targets in a different decade; observed > null means people generalize along decade categories beyond what distance-matched intervals in the model predict (observed < null: decades matter less than modelled).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = (~np.isnan(ex)).sum(axis=1)
    x = np.nan_to_num(ex[:, 0]).astype(int)
    dec = lambda v: np.minimum(v, 99) // 10
    near = (n == 1) & (np.abs(t - x) <= 9) & (t != x)
    same = near & (dec(t) == dec(x))
    diff = near & (dec(t) != dec(x))
    if not same.any() or not diff.any():
        return 0.0
    return float(r[same].mean() - r[diff].mean())
