# name: single_shared_factor_nonmultiple_contrast
# description: In single-example sets, yes-rate for targets that share a common factor >=3 with the example but are neither its multiple nor its divisor (e.g. 12 after "15", 9 after "6", 27 after "30"), minus yes-rate for targets with gcd(target, example) < 3 that are not multiples/divisors; observed > null means people generalize to factor-kin of a lone example beyond pure multiples, which the model (multiples route only) under-produces.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",").to_numpy()
    e = pd.to_numeric(s.str.split(",").str[0], errors="coerce").to_numpy(dtype=float)
    e = np.where(np.isnan(e) | (e < 1), 1.0, e).astype(np.int64)
    t = df["target"].astype(int).to_numpy().astype(np.int64)
    g = np.gcd(t, e)
    mult = (t % e == 0)
    div = (e % t == 0)
    kin = (g >= 3) & ~mult & ~div
    other = (g < 3) & ~mult & ~div
    r = df["rating"].to_numpy(dtype=float)
    a, b = single & kin, single & other
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
