# name: common_divisor_nodigit_contrast
# description: In multi-example sets whose examples' greatest common divisor is >=3, among targets sharing no example's final digit, yes-rate for targets divisible by that gcd minus for non-divisible targets; observed > null means people infer an arithmetic 'multiples of g' rule more strongly than the model's per-example coprimality penalty captures, observed < null means less.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    n = (~np.isnan(ex)).sum(1)
    exi = np.where(np.isnan(ex), 0, ex).astype(int)
    g = np.gcd.reduce(exi, axis=1)
    dig = np.mod(ex, 10)
    anydig = np.nansum((dig == np.mod(t, 10)[:, None]).astype(float), 1) > 0
    base = (n >= 2) & (g >= 3) & ~anydig
    div = np.mod(t, np.maximum(g, 1)) == 0
    r = df["rating"].to_numpy().astype(float)
    a = base & div; b = base & ~div
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
