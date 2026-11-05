# name: two_example_parity_gap_in_range
# description: On two-example sets with same-parity examples but different last digits, yes rate for in-range non-example targets of the examples' parity minus the rate for opposite-parity in-range targets; observed above null means the model under-uses parity from a pair (people restrict to the shared parity more), below null means it over-uses it.
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
    same_u = np.nanmax(U, axis=1) == np.nanmin(U, axis=1)
    P = E % 2
    same_p = np.nanmax(P, axis=1) == np.nanmin(P, axis=1)
    base = (n == 2) & same_p & ~same_u & ~is_ex & (t >= lo) & (t <= hi)
    return rate(base & (t % 2 == E[:, 0] % 2)) - rate(base & (t % 2 != E[:, 0] % 2))
