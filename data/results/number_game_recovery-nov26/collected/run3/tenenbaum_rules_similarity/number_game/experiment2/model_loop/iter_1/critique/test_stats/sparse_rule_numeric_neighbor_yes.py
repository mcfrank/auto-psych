# name: sparse_rule_numeric_neighbor_yes
# description: For sets of >=2 examples that all fall in one sparse rule (powers of 2, squares, powers of 3, or cubes), yes rate on targets in none of those sparse rules but within 5 of some example; observed > null means people retain numeric-similarity generalisation around rule-like examples that the model (whose rule posterior dominates and rejects non-members) suppresses, observed < null means they reject near neighbours even more.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    valid = ~np.isnan(E)
    rules = [np.array([1, 2, 4, 8, 16, 32, 64]), np.arange(1, 11) ** 2, np.array([1, 3, 9, 27, 81]), np.array([1, 8, 27, 64])]
    any_rule = np.zeros(len(t), bool)
    t_in_any = np.zeros(len(t), bool)
    for R in rules:
        all_in = np.all(np.isin(E, R) | ~valid, axis=1)
        any_rule |= all_in
        t_in_any |= np.isin(t, R)
    near = np.nanmin(np.abs(E - t[:, None]), axis=1)
    m = (n >= 2) & any_rule & ~t_in_any & (near >= 1) & (near <= 5)
    return float(r[m].mean()) if m.sum() else 0.0
