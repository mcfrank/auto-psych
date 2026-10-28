# name: parity_match_outside_span_gap
# description: In sets of >=2 whose examples all share a parity, yes-rate for targets outside the span with the same parity minus opposite parity; observed above null means people use the parity rule beyond the range more than the model (which favors ranges), below null means the model over-weights parity rules.
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    last = X[np.arange(len(X)), n - 1]
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    par = np.where(np.isnan(X), np.nan, np.mod(X, 2))
    allsame = (np.nanmax(par, axis=1) == np.nanmin(par, axis=1))
    sp = np.nanmin(par, axis=1)
    out = (t < mn) | (t > mx)
    g = (n >= 2) & allsame & out
    same = g & (np.mod(t, 2) == sp)
    diff = g & (np.mod(t, 2) != sp)
    if same.sum() == 0 or diff.sum() == 0:
        return 0.0
    return float(r[same].mean() - r[diff].mean())
