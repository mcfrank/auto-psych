# name: common_factor_sets_far_multiples
# description: For 2+ example sets whose examples share a common factor g >= 3 (excluding sets all of one last digit and all-powers-of-two sets), yes-rate for non-example multiples of g lying outside the set's range; observed > null means people extend a multiples-of-g rule beyond the examples more than the model predicts (observed < null: less).
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[codes]
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    inset = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    Ei = np.nan_to_num(E, nan=0).astype(np.int64)
    g = np.gcd.reduce(Ei, axis=1)
    ld = np.mod(E, 10)
    samedig = np.nanmin(ld, axis=1) == np.nanmax(ld, axis=1)
    P = np.array([2, 4, 8, 16, 32, 64], dtype=float)
    allp = np.all(np.isin(E, P) | np.isnan(E), axis=1)
    elig = (n >= 2) & (g >= 3) & ~samedig & ~allp
    gs = np.where(g > 0, g, 1)
    m = elig & ~inset & ((t < lo) | (t > hi)) & (np.mod(t.astype(np.int64), gs) == 0)
    return float(y[m].mean()) if m.any() else 0.0
