# name: parity_mismatch_yes_rate
# description: Among sets of 2+ examples that all share one parity, the yes-rate for targets of the opposite parity; observed > model means people use parity less strictly (generalize across parity more) than the model predicts, observed < model means they reject parity violators more sharply.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    par = np.where(valid, np.mod(np.nan_to_num(ex), 2), np.nan)
    allodd = (np.nansum(par, axis=1) == n)
    alleven = (np.nansum(par, axis=1) == 0)
    t = df["target"].astype(int).to_numpy()
    m = (n >= 2) & ((allodd & (t % 2 == 0)) | (alleven & (t % 2 == 1)))
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy(dtype=float)[m].mean())
