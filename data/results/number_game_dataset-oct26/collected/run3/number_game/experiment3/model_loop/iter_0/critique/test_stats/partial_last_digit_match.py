# name: partial_last_digit_match
# description: For sets of 2+ examples whose last digits are NOT all equal, among targets not in the set and outside the set's range, yes-rate for targets sharing a last digit with at least one example minus those sharing none; observed > null means people use partial feature (digit) similarity that the all-or-nothing rule model misses.
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    ex = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    E = ex[codes]  # (rows, 4) examples, NaN-padded, in shown order
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    ld = np.mod(E, 10)
    mixed = (n >= 2) & (np.nanmin(ld, axis=1) != np.nanmax(ld, axis=1))
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    inset = np.any(E == t[:, None], axis=1)
    elig = mixed & ~inset & ((t < lo) | (t > hi))
    share = np.any(ld == np.mod(t, 10)[:, None], axis=1)
    a = elig & share; b = elig & ~share
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
