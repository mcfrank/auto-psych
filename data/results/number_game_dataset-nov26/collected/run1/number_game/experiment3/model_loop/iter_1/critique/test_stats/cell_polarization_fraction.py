# name: cell_polarization_fraction
# description: Fraction of (set, target) cells (9 answers each) whose yes-rate is <=1/9 or >=8/9; observed > null means people's answers are more consensual/all-or-none per item (deterministic rule use) than the model's graded probabilities produce, observed < null means more item-level ambivalence.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t1 = df["target"].astype(int).to_numpy().astype(float)
    t = t1[:, None]
    pres = ~np.isnan(ex)
    n = pres.sum(1)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    r = df["rating"].to_numpy(dtype=float)
    d_all = np.abs(t - e)
    dist = d_all.min(1)
    dig = ((np.mod(t, 10) == np.mod(e1, 10)) & pres).any(1)
    isex = (t == e).any(1)
    mult = (pres & (e1 > 1) & (np.mod(t, e1) == 0) & (t != e)).any(1)
    divi = (pres & (np.mod(e1, t) == 0) & (t != e)).any(1)
    kin = (pres & (np.gcd(t.astype(int), e1.astype(int)) >= 3)).any(1)
    emin = np.where(pres, ex, 1e9).min(1)
    emax = np.where(pres, ex, -1e9).max(1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    m = pd.Series(r).groupby(key.to_numpy()).mean().to_numpy()
    return float(((m <= 1.0 / 9 + 1e-9) | (m >= 8.0 / 9 - 1e-9)).mean())
