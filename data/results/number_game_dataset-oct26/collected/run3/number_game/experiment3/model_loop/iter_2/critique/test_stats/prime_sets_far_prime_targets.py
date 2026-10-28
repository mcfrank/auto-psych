# name: prime_sets_far_prime_targets
# description: For 2+ example sets whose examples are all primes, yes-rate for prime non-example targets lying more than 10 outside the set's [min,max] range; observed > null means people extend a "primes"/"odd" concept beyond the examples' range more than the model (whose tight intervals out-compete the 26-member prime rule) predicts (observed < null: less).
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[codes]
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    inset = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    PR = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97], dtype=float)
    allp = (n >= 2) & np.all(np.isin(E, PR) | np.isnan(E), axis=1)
    far = (t < lo - 10) | (t > hi + 10)
    m = allp & ~inset & far & np.isin(t, PR)
    return float(y[m].mean()) if m.any() else 0.0
