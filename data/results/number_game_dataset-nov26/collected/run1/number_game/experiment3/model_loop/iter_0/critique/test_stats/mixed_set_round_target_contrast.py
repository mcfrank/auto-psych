# name: mixed_set_round_target_contrast
# description: In sets of >=2 examples where some but not all examples end in 0 or 5 (e.g. 15,11; 6,25), yes-rate for multiple-of-10 targets (>=10 away, no shared final digit) minus yes-rate for non-round unrelated targets (>=10 away, no shared digit, not multiple/divisor/gcd>=3 kin); observed < null means the model's round-number kinship route over-generalizes from a lone 5-ending example inside a mixed set, observed > null means it under-credits round targets there.
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
    end5 = (np.mod(e1, 5) == 0) & pres
    mixed = (n >= 2) & end5.any(1) & ~(end5 | ~pres).all(1)
    base = mixed & (dist >= 10) & ~dig
    roundt = np.mod(t1, 10) == 0
    return rate(base & roundt) - rate(base & ~roundt & ~mult & ~divi & ~kin)
