# name: shared_digit_3plus_nonmatching_same_parity
# description: On 3-4 example sets whose examples all share a last digit, yes rate for non-example targets with a different last digit but the same parity as the examples; observed below null means the model (even with explaining away) still leaks too much generalization to same-parity / contains-digit numbers when an ends-in-digit rule fits.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(E)
    n = valid.sum(axis=1)
    Ez = np.where(valid, E, 0.0)
    lo = np.where(valid, E, np.inf).min(axis=1)
    hi = np.where(valid, E, -np.inf).max(axis=1)
    is_ex = (valid & (E == t[:, None])).any(axis=1)
    odd = (np.where(valid, Ez % 2, 1) == 1).all(axis=1)
    even = (np.where(valid, Ez % 2, 0) == 0).all(axis=1)
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    ld = np.where(valid, Ez % 10, np.nan)
    first = ld[:, 0]
    shared = (np.where(valid, ld == first[:, None], True)).all(axis=1)
    same_par = np.where(odd, t % 2 == 1, np.where(even, t % 2 == 0, False))
    m = (n >= 3) & shared & ~is_ex & ((t % 10) != first) & same_par
    return rate(m)
