# name: single_example_shared_factor_contrast
# description: In single-example sets, yes-rate for targets sharing a common factor >=3 with the example (gcd(target, example) >= 3, e.g. 14/49/98 for "7", 30/100 for "5") minus yes-rate for other targets; observed > null means people infer a 'multiples/factors of the example' rule from one example, which the model (no factor term for single examples) under-produces, observed < null means less.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    single = ~sets.str.contains(",").to_numpy()
    e = pd.to_numeric(sets.where(single, "0"), errors="coerce").fillna(0).astype(int).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    g = np.gcd(e, t)
    base = single & (t != e)
    a = base & (g >= 3)
    b = base & (g < 3)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
