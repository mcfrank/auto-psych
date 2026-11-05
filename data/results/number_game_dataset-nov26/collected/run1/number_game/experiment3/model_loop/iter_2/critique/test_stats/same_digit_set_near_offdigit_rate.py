# name: same_digit_set_near_offdigit_rate
# description: In multi-example sets whose examples all share one final digit, yes-rate for targets 1-5 away from the nearest example that do NOT share that digit; observed > null means people keep accepting close off-rule neighbours (magnitude survives the digit rule) more than the model's decisive slope allows, observed < null means people reject near rule-violators more sharply than predicted.
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
    m = alldig & (dmin >= 1) & (dmin <= 5) & ~sharedig
    return rate(m)
