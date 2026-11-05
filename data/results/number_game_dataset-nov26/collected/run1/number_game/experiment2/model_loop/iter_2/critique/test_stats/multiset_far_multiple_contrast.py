# name: multiset_far_multiple_contrast
# description: In sets with >=2 examples, yes-rate for targets that are a multiple (not equal) of some example >1 yet >=10 from every example and sharing no example's final digit, minus yes-rate for equally far non-matching non-multiple targets; observed > null means people use the multiples route in multi-example sets more than the model's fixed additive route predicts, observed < null means the model over-credits multiples once several examples are shown.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)[:, None]
    pres = ~np.isnan(ex)
    e = np.where(pres, ex, 1e9)
    n = pres.sum(1)
    far = (np.abs(t - e) >= 10).all(1)
    nodig = ~((np.mod(t, 10) == np.mod(e, 10)) & pres).any(1)
    mult = (pres & (e > 1) & (np.mod(t, np.where(pres, ex, 1.0)) == 0) & (t != e)).any(1)
    base = (n >= 2) & far & nodig
    r = df["rating"].to_numpy(dtype=float)
    a, b = base & mult, base & ~mult
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
