# name: opp_parity_just_outside_yes
# description: In sets of >=2 whose examples all share a parity, yes-rate for opposite-parity targets 1-3 outside the example span; observed below null means the model over-extends ranges to opposite-parity neighbours (people extrapolate by parity), above null means people extend magnitude ranges past the edge more than the model.
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
    dist = np.maximum(mn - t, t - mx)
    m = (n >= 2) & allsame & (dist >= 1) & (dist <= 3) & (np.mod(t, 2) != sp)
    return float(r[m].mean()) if m.any() else 0.0
