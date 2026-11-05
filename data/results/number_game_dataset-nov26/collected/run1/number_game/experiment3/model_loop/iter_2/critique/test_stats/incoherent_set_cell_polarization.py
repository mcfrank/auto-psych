# name: incoherent_set_cell_polarization
# description: Among (set, target) cells from single-example sets and incoherent multi-example sets (no shared final digit, parity or multiple-of-5 among examples), fraction of cells whose yes-rate is <=1/9 or >=8/9; observed > null means people answer all-or-none even without a suspicious coincidence (decisiveness is not confined to coherent sets as the model assumes), observed < null means the model is too polarized there.
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
    key = (df["number_set"].astype(str) + "|" + df["target"].astype(str)).to_numpy()
    m = pd.Series(r).groupby(key).mean()
    c = pd.Series(coh).groupby(key).first()
    m = m[~c.reindex(m.index).to_numpy()]
    if len(m) == 0:
        return 0.0
    return float(((m <= 1.0 / 9 + 1e-9) | (m >= 8.0 / 9 - 1e-9)).mean())
