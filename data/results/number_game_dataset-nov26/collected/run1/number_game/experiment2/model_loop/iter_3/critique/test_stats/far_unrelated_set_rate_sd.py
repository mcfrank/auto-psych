# name: far_unrelated_set_rate_sd
# description: Standard deviation across sets (with >=9 such trials) of the yes-rate for "unrelated" targets (>=15 from every example, sharing no example's final digit, not a multiple/divisor/factor-kin of any example); observed > null means sets differ in how broadly people generalize to unrelated numbers more than the model's stimulus-driven predictions allow (set-level breadth heterogeneity it under-produces), observed < null means less.
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
    nodig = ~((np.mod(t, 10) == np.mod(e, 10)) & pres).any(1)
    mult = (pres & (e1 > 1) & (np.mod(t, e1) == 0) & (t != e)).any(1)
    divi = (pres & (np.mod(e1, t) == 0) & (t != e)).any(1)
    g = np.gcd(t.astype(int), e1.astype(int))
    kin = (pres & (g >= 3)).any(1)

    m = (d >= 15) & nodig & ~mult & ~divi & ~kin
    s = pd.DataFrame({"s": df["number_set"].to_numpy()[m], "r": r[m]})
    gr = s.groupby("s")["r"].agg(["mean", "size"])
    gr = gr[gr["size"] >= 9]
    if len(gr) < 2:
        return 0.0
    return float(gr["mean"].std())
