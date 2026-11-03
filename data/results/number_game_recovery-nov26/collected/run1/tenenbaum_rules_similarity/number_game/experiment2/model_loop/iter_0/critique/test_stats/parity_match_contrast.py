# name: parity_match_contrast
# description: For multi-example sets whose examples all share parity, yes rate on parity-matching targets minus parity-mismatching targets; observed above null means people use parity more than the model (model under-weights odd/even), below null means the model over-weights it.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    par = np.where(np.isnan(ex), np.nan, ex % 2)
    pmin = np.nanmin(par, axis=1); pmax = np.nanmax(par, axis=1)
    same = (pmin == pmax) & (n >= 2)
    t = df["target"].astype(int).to_numpy()
    match = (t % 2) == pmin
    y = df["rating"].astype(float).to_numpy()
    a = same & match; b = same & ~match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
