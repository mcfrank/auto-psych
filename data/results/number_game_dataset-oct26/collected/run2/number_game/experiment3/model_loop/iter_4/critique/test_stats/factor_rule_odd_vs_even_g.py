# name: factor_rule_odd_vs_even_g
# description: Multiple-minus-non-multiple yes-rate gap (non-example targets) for sets whose GCD g is even and >= 4 minus the same gap for sets whose g is odd and >= 3; observed > null means people pick up "multiples of g" more readily for even g (4, 8, 12...) than odd g (3, 9...), which the model's log(g)-scaled rule does not capture (it over-predicts odd-g and under-predicts even-g rules).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).values
    t = df["target"].astype(int).values
    k = (ex > 0).sum(axis=1)
    g = np.gcd.reduce(ex, axis=1)
    isex = (ex == t[:, None]).any(axis=1)
    r = df["rating"].values.astype(float)
    base = (k >= 2) & ~isex & (g >= 3)
    mult = (t % np.maximum(g, 1)) == 0
    def gap(s):
        a = s & mult; b = s & ~mult
        if a.sum() == 0 or b.sum() == 0:
            return np.nan
        return r[a].mean() - r[b].mean()
    v = gap(base & (g % 2 == 0)) - gap(base & (g % 2 == 1))
    return float(v) if np.isfinite(v) else 0.0
