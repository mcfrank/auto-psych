# name: power2_sets_even_nonpower_targets
# description: For 2+ example sets whose examples are all powers of two (2..64), yes-rate for even non-example targets that are NOT powers of two; observed > null means people generalize powers-of-two sets to even numbers / multiples more broadly than the model's size-principle posterior (dominated by the powers-of-two rule) predicts (observed < null: humans stricter).
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[codes]
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    inset = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    P = np.array([2, 4, 8, 16, 32, 64], dtype=float)
    isp = np.isin(E, P) | np.isnan(E)
    allp = (n >= 2) & np.all(isp, axis=1)
    m = allp & ~inset & (np.mod(t, 2) == 0) & ~np.isin(t, P)
    return float(y[m].mean()) if m.any() else 0.0
