# name: math_rule_far_member_yes
# description: In sets of >= 2 examples that are all powers of 2, all perfect squares, or all perfect cubes, yes rate for non-example targets belonging to that same mathematical family and >= 5 from every example; observed > null means people extend mathematical rules to distant members more than modelled, observed < null means they prefer range/similarity over the math rule more than modelled.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    d = dist.min(axis=1)
    fams = [np.array([2, 4, 8, 16, 32, 64]), np.array([k * k for k in range(1, 11)]), np.array([1, 8, 27, 64])]
    m = np.zeros(len(t), dtype=bool)
    for f in fams:
        allin = (np.isin(ex, f) | ~valid).all(axis=1) & (n >= 2)
        m |= allin & np.isin(t, f) & (d >= 5)
    return float(y[m].mean()) if m.sum() > 0 else 0.0
