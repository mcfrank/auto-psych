# name: round_target_in_nonround_even_set
# description: In sets whose examples are all even and none ends in 0 or 5, yes-rate for multiple-of-10 targets minus yes-rate for other even targets sharing no example's final digit; observed < null means people actively disfavour round numbers when the set is not round (the model treats them as ordinary non-matching evens), observed > null means a general round-number bonus the model lacks.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t1 = df["target"].astype(int).to_numpy().astype(float)
    t = t1[:, None]
    pres = ~np.isnan(ex)
    n = pres.sum(1)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    r = df["rating"].to_numpy(dtype=float)
    d = np.abs(t - e).min(1)
    digmatch = ((np.mod(t, 10) == np.mod(e1, 10)) & pres).any(1)
    def diff(a, b):
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return float(r[a].mean() - r[b].mean())
    alleven = (~pres | (np.mod(e1, 2) == 0)).all(1)
    no05 = ~((np.mod(e1, 5) == 0) & pres).any(1)
    s = alleven & no05
    rnd = np.mod(t1, 10) == 0
    ev = np.mod(t1, 2) == 0
    return diff(s & rnd, s & ev & ~rnd & ~digmatch)
