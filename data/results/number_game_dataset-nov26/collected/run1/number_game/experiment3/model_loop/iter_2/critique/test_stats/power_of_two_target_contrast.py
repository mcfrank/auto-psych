# name: power_of_two_target_contrast
# description: In multi-example sets made only of powers of 2 (e.g. 16,8 / 32,4,16), yes-rate for non-example power-of-2 targets minus yes-rate for other even non-example targets; observed > null means people infer a powers-of-2 rule the model (multiples route + set-gcd only) does not capture, observed < null means the model over-favours power-of-2 targets.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy(dtype=float)
    pres = ~np.isnan(ex)
    n = pres.sum(1)
    e1 = np.where(pres, ex, 1.0).astype(int)
    tt = t[:, None]
    d = np.abs(tt - np.where(pres, ex, 1e9))
    dmin = d.min(1)
    dig = np.where(pres, np.mod(e1, 10), -1)
    par = np.where(pres, np.mod(e1, 2), -1)
    alldig = (np.where(pres, dig, dig[:, [0]]) == dig[:, [0]]).all(1) & (n > 1)
    allpar = (np.where(pres, par, par[:, [0]]) == par[:, [0]]).all(1) & (n > 1)
    all5 = (np.where(pres, np.mod(e1, 5), 0) == 0).all(1) & (n > 1)
    coh = alldig | allpar | all5
    sharedig = ((np.mod(tt, 10) == dig) & pres).any(1)
    mult = ((np.mod(tt, e1) == 0) & pres & (e1 > 1) & (tt != e1)).any(1)
    divi = ((np.mod(e1, tt) == 0) & pres & (tt != e1)).any(1)
    kin = ((np.gcd(tt, e1) >= 3) & pres).any(1)
    emin = np.where(pres, ex, 1e9).min(1)
    emax = np.where(pres, ex, -1e9).max(1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    p2 = np.isin(e1, [1, 2, 4, 8, 16, 32, 64]) | ~pres
    sp2 = p2.all(1) & (n > 1)
    tp2 = np.isin(t, [1, 2, 4, 8, 16, 32, 64])
    notex = dmin > 0
    return rate(sp2 & tp2 & notex) - rate(sp2 & ~tp2 & (np.mod(t, 2) == 0) & notex)
