# name: opposite_parity_yes_in_parity_sets
# description: Yes rate for targets of the opposite parity to the examples, among sets of >= 2 examples all sharing one parity; observed < null means the model's interval/noise mass over-predicts yes to parity-violating targets, observed > null means it under-predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    par = np.where(np.isnan(ex), np.nan, np.mod(ex, 2))
    same = (np.nanmax(par, axis=1) == np.nanmin(par, axis=1)) & (n >= 2)
    m = same & (np.mod(t, 2) != np.nanmax(par, axis=1))
    return float(y[m].mean()) if m.sum() else 0.0
