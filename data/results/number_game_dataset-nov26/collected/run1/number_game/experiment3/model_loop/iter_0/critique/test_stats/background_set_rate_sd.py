# name: background_set_rate_sd
# description: SD across number sets (with >=5 such trials) of the yes-rate for unrelated background targets (>=10 from every example, no shared final digit, not a multiple/divisor/gcd>=3 kin of any example); observed > null means sets differ in their baseline willingness to say yes far more than the model's shared criterion allows (set-specific strictness), observed < null the reverse.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t1 = df["target"].astype(int).to_numpy().astype(float)
    t = t1[:, None]
    pres = ~np.isnan(ex)
    n = pres.sum(1)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    r = df["rating"].to_numpy(dtype=float)
    dist = np.abs(t - e).min(1)
    dig = ((np.mod(t, 10) == np.mod(e1, 10)) & pres).any(1)
    isex = (t == e).any(1)
    mult = (pres & (e1 > 1) & (np.mod(t, e1) == 0) & (t != e)).any(1)
    divi = (pres & (np.mod(e1, t) == 0) & (t != e)).any(1)
    kin = (pres & (np.gcd(t.astype(int), e1.astype(int)) >= 3)).any(1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    bg = (dist >= 10) & ~dig & ~mult & ~divi & ~kin
    s = pd.DataFrame({"s": df["number_set"].to_numpy()[bg], "r": r[bg]})
    g = s.groupby("s")["r"].agg(["mean", "count"])
    g = g[g["count"] >= 5]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
