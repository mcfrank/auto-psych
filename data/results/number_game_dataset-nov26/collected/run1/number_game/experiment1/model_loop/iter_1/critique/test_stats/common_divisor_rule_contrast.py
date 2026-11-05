# name: common_divisor_rule_contrast
# description: In size>=2 sets whose GCD is >=3 and not 10, yes rate for targets divisible by the GCD minus not divisible (distance>5 from examples); observed above null means the model under-produces multiples-rule generalisation beyond final-digit similarity.
def test_statistic(df):
    S = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(S), axis=1)
    Si = np.nan_to_num(S, nan=0.0).astype(int)
    dist = np.nanmin(np.abs(S - t[:, None]), axis=1)
    smin = np.nanmin(S, axis=1); smax = np.nanmax(S, axis=1)
    def m(mask):
        return float(np.mean(r[mask])) if np.sum(mask) > 0 else 0.0
    g = np.gcd.reduce(Si, axis=1)
    sel = (n >= 2) & (g >= 3) & (g != 10) & (dist > 5)
    gg = np.where(g > 0, g, 1)
    div = (t % gg) == 0
    return m(sel & div) - m(sel & ~div)
