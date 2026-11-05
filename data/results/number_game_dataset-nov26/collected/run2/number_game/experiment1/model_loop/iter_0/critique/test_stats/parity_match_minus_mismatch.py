# name: parity_match_minus_mismatch
# description: Among uniform-parity sets, yes-rate for parity-matching targets minus yes-rate for parity-mismatching targets; observed < null means humans weight parity less than the model (model over-relies on odd/even rules), observed > null means more.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    par = np.mod(ex, 2)
    n = np.sum(~np.isnan(ex), axis=1)
    s = np.nansum(par, axis=1)
    uniform = (s == 0) | (s == n)
    setpar = (s == n).astype(float)
    tp = np.mod(df["target"].astype(int).to_numpy(), 2)
    y = df["rating"].to_numpy().astype(float)
    a = uniform & (tp == setpar)
    b = uniform & (tp != setpar)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
