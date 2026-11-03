# name: unrelated_target_floor
# description: Yes rate among targets more than 10 from every example, sharing no last digit with any example, and not divisible by the examples' gcd when that gcd is >=2 (or any target when gcd is 1); observed above null means a larger lapse/noise floor of yes to unrelated numbers than the model predicts, below null means a smaller one.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    nearest = dist.min(axis=1)
    nonmember = nearest > 0
    ei = np.where(valid, ex, 0).astype(np.int64)
    g = np.gcd.reduce(ei, axis=1)
    ti = t.astype(np.int64)
    div = (g >= 2) & (np.mod(ti, np.maximum(g, 1)) == 0)
    same_ld = (valid & (np.mod(np.nan_to_num(ex), 10) == np.mod(t, 10)[:, None])).any(axis=1)
    m = nonmember & (nearest > 10) & ~same_ld & ~div
    return float(r[m].mean()) if m.any() else 0.0
