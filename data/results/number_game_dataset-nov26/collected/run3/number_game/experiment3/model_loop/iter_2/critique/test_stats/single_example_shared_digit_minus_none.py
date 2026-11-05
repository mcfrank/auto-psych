# name: single_example_shared_digit_minus_none
# description: On one-example sets, yes rate for non-example targets sharing the example's units or tens digit minus yes rate for targets sharing neither digit; observed above null means the model's single-example discount of digit rules is too strong (people generalize by shared digits from one example more than predicted), below null means too weak.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    U = E % 10; tu = t % 10
    def rate(m):
        return float(y[m].mean()) if m.any() else 0.0
    tt = t // 10; T0 = E[:, 0] // 10
    share = (tu == U[:, 0]) | ((tt == T0) & (t < 100) & (E[:, 0] < 100))
    base = (n == 1) & ~is_ex
    return rate(base & share) - rate(base & ~share)
