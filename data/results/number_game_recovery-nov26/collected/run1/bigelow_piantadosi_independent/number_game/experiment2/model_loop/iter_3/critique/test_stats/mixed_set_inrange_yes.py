# name: mixed_set_inrange_yes
# description: Among sets of >= 2 examples that share neither parity nor final digit, yes rate for non-example targets lying strictly between the smallest and largest example; observed > null means people fill in the examples' range (interval generalisation for rule-less sets) more than the model, observed < null less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    par = np.mod(ex, 2)
    ld = np.mod(ex, 10)
    mixed = (n >= 2) & (np.nanmax(par, axis=1) != np.nanmin(par, axis=1)) & (np.nanmax(ld, axis=1) != np.nanmin(ld, axis=1))
    m = mixed & (d > 0) & (t > np.nanmin(ex, axis=1)) & (t < np.nanmax(ex, axis=1))
    return float(y[m].mean()) if m.sum() else 0.0
