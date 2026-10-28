# name: parity_match_in_range
# description: For sets of 2+ examples that all share one parity, among targets strictly inside the set's range and not in the set, yes-rate for parity-matching targets minus parity-mismatching ones; observed > null means people weight parity within an interval more than the model does (observed < null: less).
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    ex = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    E = ex[codes]  # (rows, 4) examples, NaN-padded, in shown order
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    par = np.mod(E, 2)
    pmin = np.nanmin(par, axis=1); pmax = np.nanmax(par, axis=1)
    same = (n >= 2) & (pmin == pmax)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    inset = np.any(E == t[:, None], axis=1)
    inr = same & (t > lo) & (t < hi) & ~inset
    match = inr & (np.mod(t, 2) == pmin)
    mis = inr & (np.mod(t, 2) != pmin)
    if match.sum() == 0 or mis.sum() == 0:
        return 0.0
    return float(y[match].mean() - y[mis].mean())
