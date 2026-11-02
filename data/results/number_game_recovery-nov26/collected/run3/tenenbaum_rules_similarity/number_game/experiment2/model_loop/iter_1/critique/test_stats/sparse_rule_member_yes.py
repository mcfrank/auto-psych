# name: sparse_rule_member_yes
# description: For sets of >=2 examples that all fall in a sparse rule (powers of 2, squares, powers of 3, or cubes), yes rate on non-example targets that belong to a sparse rule containing all the examples; observed < null means people extend the rule to new members less than the model's rule posterior predicts (observed > null: more).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    valid = ~np.isnan(E)
    in_set = np.any(E == t[:, None], axis=1)
    rules = [np.array([1, 2, 4, 8, 16, 32, 64]), np.arange(1, 11) ** 2, np.array([1, 3, 9, 27, 81]), np.array([1, 8, 27, 64])]
    hit = np.zeros(len(t), bool)
    for R in rules:
        hit |= np.all(np.isin(E, R) | ~valid, axis=1) & np.isin(t, R)
    m = (n >= 2) & hit & ~in_set
    return float(r[m].mean()) if m.sum() else 0.0
