# name: parity_outlier_far_majority_yes
# description: In sets of >= 3 examples with exactly one example of the minority parity, yes rate for non-example targets of the majority parity more than 10 from every example; observed > null means people extend the majority parity rule to distant numbers despite the exception more than the model, observed < null means the exception suppresses rule extension more than modelled.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    par = np.mod(ex, 2)
    odd = np.nansum(par == 1, axis=1)
    even = np.nansum(par == 0, axis=1)
    one_out = (n >= 3) & ((odd == 1) | (even == 1))
    maj = np.where(odd > even, 1, 0)
    m = one_out & (np.mod(t, 2) == maj) & (d > 10)
    return float(y[m].mean()) if m.sum() else 0.0
