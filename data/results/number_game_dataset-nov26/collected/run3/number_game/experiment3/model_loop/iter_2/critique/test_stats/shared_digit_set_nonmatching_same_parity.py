# name: shared_digit_set_nonmatching_same_parity
# description: On 3-4 example sets whose examples all share a last digit, yes rate for non-example targets with a different last digit but the same parity as the examples; observed below null means the model leaks too much generalization to same-parity numbers (via odd/even, contains-digit, prime rules) when an ends-in-digit rule fits.
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
    m = (n >= 3) & same_u & ~is_ex & (tu != U[:, 0]) & (t % 2 == E[:, 0] % 2)
    return rate(m)
