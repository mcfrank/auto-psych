# name: odd_vs_even_set_parity_violation
# description: In parity-homogeneous sets of >=3 examples (excluding sets whose examples all end in 0 or 5), yes-rate for opposite-parity targets (>=10 away, no shared final digit) in all-odd sets minus that in all-even sets; observed > null means people treat 'odd' as a much weaker/less salient rule than 'even' (the model's parity attention is symmetric), observed < null the reverse.
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
    par = np.where(pres, np.mod(e1, 2), np.nan)
    pmax = np.nanmax(par, 1)
    pmin = np.nanmin(par, 1)
    round_set = ((np.mod(e1, 5) == 0) | ~pres).all(1)
    homog = (pmax == pmin) & (n >= 3) & ~round_set
    viol = (np.mod(t1, 2) != pmax) & (dist >= 10) & ~dig
    return rate(homog & (pmax == 1) & viol) - rate(homog & (pmax == 0) & viol)
