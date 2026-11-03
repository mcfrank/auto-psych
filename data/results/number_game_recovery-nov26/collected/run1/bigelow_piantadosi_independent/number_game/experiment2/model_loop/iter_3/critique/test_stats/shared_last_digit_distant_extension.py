# name: shared_last_digit_distant_extension
# description: Among sets of >= 2 examples all ending in the same digit, yes rate for non-example targets ending in that digit and more than 5 from every example; observed > null means people extend "ends in d" rules to distant numbers more than the model, observed < null less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    ld = np.where(np.isnan(ex), np.nan, np.mod(ex, 10))
    same = (np.nanmax(ld, axis=1) == np.nanmin(ld, axis=1)) & (n >= 2)
    m = same & (np.mod(t, 10) == np.nanmax(ld, axis=1)) & (d > 5)
    return float(y[m].mean()) if m.sum() else 0.0
