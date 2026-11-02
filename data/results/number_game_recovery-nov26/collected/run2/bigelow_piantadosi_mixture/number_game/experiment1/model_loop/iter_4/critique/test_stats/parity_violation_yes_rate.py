# name: parity_violation_yes_rate
# description: Among sets of 2+ examples that are all even or all odd, yes rate on targets of the opposite parity; observed above null means people respect parity less than the model (too few false alarms in the model), below null means the model over-generalizes across parity.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    fin = np.isfinite(ex)
    n = fin.sum(axis=1)
    par = np.where(fin, np.mod(np.nan_to_num(ex), 2), np.nan)
    ev = np.nansum(par == 0, axis=1); od = np.nansum(par == 1, axis=1)
    alle = (n >= 2) & (ev == n); allo = (n >= 2) & (od == n)
    m = (alle & (t % 2 == 1)) | (allo & (t % 2 == 0))
    return float(y[m].mean()) if m.sum() > 0 else 0.0
