# name: parity_match_effect
# description: Among multi-example sets whose examples all share a parity, and non-example targets, yes-rate when the target shares that parity minus when it does not; observed below the null means the model over-weights parity (even/odd rules) relative to humans, above means it under-weights it.
def test_statistic(df):
    arr = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = np.sum(~np.isnan(arr), axis=1)
    t = df["target"].astype(float).values
    par = np.mod(arr, 2)
    mx = np.nanmax(par, axis=1)
    mn = np.nanmin(par, axis=1)
    same = (mx == mn) & (n >= 2)
    dist = np.nanmin(np.abs(arr - t[:, None]), axis=1)
    keep = same & (dist > 0)
    match = np.mod(t, 2) == mx
    r = df["rating"].values.astype(float)
    a = keep & match
    b = keep & ~match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
