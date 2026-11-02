# name: parity_even_minus_odd_asymmetry
# description: Parity-match effect (yes rate for same-parity minus opposite-parity non-example targets) in all-even sets minus the same effect in all-odd sets (>=2 examples); t_observed differing from null_mean means people treat "even" and "odd" rules asymmetrically, which the model's single shared parity prior cannot capture (above = even rule more favoured than the model allows).
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    v = ~np.isnan(s); n = v.sum(axis=1)
    t = df["target"].astype(int).to_numpy()
    par = np.where(v, s % 2, np.nan)
    pmin = np.nanmin(par, axis=1); pmax = np.nanmax(par, axis=1)
    is_ex = (np.where(v, s, -1) == t[:, None]).any(axis=1)
    r = df["rating"].to_numpy(dtype=float)
    base = (n >= 2) & ~is_ex & (pmin == pmax)
    tp = t % 2
    def eff(p):
        m = base & (pmin == p)
        a = m & (tp == p); b = m & (tp != p)
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return r[a].mean() - r[b].mean()
    return float(eff(0) - eff(1))
