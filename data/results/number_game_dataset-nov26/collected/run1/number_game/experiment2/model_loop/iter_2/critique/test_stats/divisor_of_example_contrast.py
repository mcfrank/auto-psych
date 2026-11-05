# name: divisor_of_example_contrast
# description: Yes-rate for targets (>=2) that are a proper divisor of some example (e.g. 7 after 49, 8 after 64), are >=5 from every example and share no example's final digit, minus yes-rate for other such far non-matching targets that neither divide nor are multiples of any example; observed > null means people generalize downward to factors of an example (a 'divisor route' the model lacks), observed < null means less.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)[:, None]
    pres = ~np.isnan(ex)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    far = (np.abs(t - e) >= 5).all(1)
    nodig = ~((np.mod(t, 10) == np.mod(e, 10)) & pres).any(1)
    div = (pres & (t >= 2) & (t < e) & (np.mod(e1, t) == 0)).any(1)
    mult = (pres & (e > 1) & (np.mod(t, e1) == 0) & (t != e)).any(1)
    base = far & nodig
    r = df["rating"].to_numpy(dtype=float)
    a, b = base & div, base & ~div & ~mult
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
