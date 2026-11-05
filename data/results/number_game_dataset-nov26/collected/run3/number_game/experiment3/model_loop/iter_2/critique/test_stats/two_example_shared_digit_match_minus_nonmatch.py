# name: two_example_shared_digit_match_minus_nonmatch
# description: On two-example sets whose examples share a last digit, yes rate for non-example targets with that last digit minus yes rate for same-parity targets with another last digit; observed above null means people pick up the ends-in-digit rule from a pair more sharply than the model predicts, below null means less sharply.
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
    base = (n == 2) & same_u & ~is_ex
    return rate(base & (tu == U[:, 0])) - rate(base & (tu != U[:, 0]) & (t % 2 == E[:, 0] % 2))
