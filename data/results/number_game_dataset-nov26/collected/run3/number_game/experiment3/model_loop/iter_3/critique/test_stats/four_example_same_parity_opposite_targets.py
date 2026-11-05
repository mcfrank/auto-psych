# name: four_example_same_parity_opposite_targets
# description: On 4-example sets whose examples are all odd or all even, yes rate for non-example targets of the opposite parity; observed below null means the model over-generalizes across parity after four same-parity examples (people treat parity as more binding than the model).
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
    opp = np.where(odd, t % 2 == 0, np.where(even, t % 2 == 1, False))
    m = (n == 4) & (odd | even) & ~is_ex & opp
    return rate(m)
