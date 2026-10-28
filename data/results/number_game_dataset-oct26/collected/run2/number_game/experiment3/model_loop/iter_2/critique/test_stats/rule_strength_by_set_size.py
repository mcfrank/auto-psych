# name: rule_strength_by_set_size
# description: For sets whose examples share a common factor g>=3, the conforming-minus-nonconforming yes-rate gap (non-example targets) for 3-4 example sets minus that for 2-example sets; observed < null means rule confidence grows less with extra examples than the model's (k-1)*log g evidence scaling implies (observed > null: grows more).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    exv = ex.to_numpy(dtype=float)
    k = ex.notna().sum(axis=1).to_numpy()
    g = np.gcd.reduce(np.nan_to_num(exv, nan=0.0).astype(int), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    isex = (exv == t[:, None]).any(axis=1)
    conf = (t % np.where(g > 0, g, 1)) == 0
    base = (g >= 3) & ~isex
    def gap(m):
        a = base & m & conf; b = base & m & ~conf
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return y[a].mean() - y[b].mean()
    return float(gap(k >= 3) - gap(k == 2))
