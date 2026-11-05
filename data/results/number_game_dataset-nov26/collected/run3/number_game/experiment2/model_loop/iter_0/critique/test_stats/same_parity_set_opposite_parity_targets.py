# name: same_parity_set_opposite_parity_targets
# description: Yes rate for targets of the opposite parity to the examples, on sets of 3+ examples that are all odd or all even; observed below null means people respect the examples' shared parity more strictly than the model (above = less strictly).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    t = df["target"].astype(int).to_numpy()
    n_ex = np.sum(~np.isnan(ex), axis=1)
    par = np.where(np.isnan(ex), np.nan, np.mod(ex, 2))
    all_odd = np.nansum(par == 1, axis=1) == n_ex
    all_even = np.nansum(par == 0, axis=1) == n_ex
    tp = t % 2
    m = (n_ex >= 3) & ((all_odd & (tp == 0)) | (all_even & (tp == 1)))
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
