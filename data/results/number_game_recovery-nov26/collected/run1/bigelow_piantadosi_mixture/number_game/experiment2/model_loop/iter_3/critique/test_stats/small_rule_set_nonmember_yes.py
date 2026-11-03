# name: small_rule_set_nonmember_yes
# description: Sets of >=2 examples all belonging to a small arithmetic rule (squares, cubes, powers of 2/3, or powers of 2 plus 37): yes-rate for non-example targets outside every such rule the set fits; observed below null_mean means people reject off-rule numbers more sharply than the model (model's interval/lapse mass too high for rule-like sets), above means the opposite.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    isex = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    dmin = np.nanmin(np.abs(E - t[:, None]), axis=1)
    ld = np.mod(E, 10)
    rules = [[k * k for k in range(1, 11)], [1, 8, 27, 64], [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [2, 4, 8, 16, 32, 37, 64]]
    fits_any = np.zeros(len(t), dtype=bool); t_in = np.zeros(len(t), dtype=bool)
    valid = ~np.isnan(E)
    for R in rules:
        R = np.array(R, float)
        fit = np.all(np.isin(E, R) | ~valid, axis=1)
        fits_any |= fit
        t_in |= fit & np.isin(t, R)
    m = (n >= 2) & fits_any & ~t_in & ~isex
    return float(r[m].mean()) if m.sum() > 0 else 0.0
