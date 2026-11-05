# name: parity_rule_contrast
# description: In all-even sets of size>=2, yes rate for even minus odd targets at distance>10 from every example; observed above null means the model under-produces parity-rule generalisation.
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
    ev = (n >= 2) & np.all((Si % 2 == 0) | np.isnan(S), axis=1)
    far = dist > 10
    return m(ev & far & (t % 2 == 0)) - m(ev & far & (t % 2 == 1))
