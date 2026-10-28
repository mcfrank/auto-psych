# name: parity_gap_inside_minus_outside
# description: In sets of >=2 sharing a parity, (same-minus-opposite parity yes gap for non-example targets inside the span) minus (the same gap for targets outside the span); observed below null means people interpolate by magnitude but extrapolate by parity more than the model, above null means parity matters more inside the span than the model predicts.
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    par = np.where(np.isnan(X), np.nan, np.mod(X, 2))
    allsame = np.nanmax(par, axis=1) == np.nanmin(par, axis=1)
    sp = np.nanmin(par, axis=1)
    g = (n >= 2) & allsame & ~is_ex
    same = np.mod(t, 2) == sp
    ins = (t > mn) & (t < mx)
    out = (t < mn) | (t > mx)
    def gap(region):
        a = g & region & same
        b = g & region & ~same
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return r[a].mean() - r[b].mean()
    return float(gap(ins) - gap(out))
