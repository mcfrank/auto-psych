# name: square_sets_square_targets
# description: For 2+ example sets whose examples are all perfect squares (excluding sets that are all powers of two), yes-rate for perfect-square non-example targets; observed < null means people endorse the "squares" concept less than the model's square rule posterior predicts (observed > null: more).
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[codes]
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    inset = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    SQ = np.arange(1, 11, dtype=float) ** 2
    P = np.array([2, 4, 8, 16, 32, 64], dtype=float)
    alls = (n >= 2) & np.all(np.isin(E, SQ) | np.isnan(E), axis=1)
    allp = np.all(np.isin(E, P) | np.isnan(E), axis=1)
    m = alls & ~allp & ~inset & np.isin(t, SQ)
    return float(y[m].mean()) if m.any() else 0.0
