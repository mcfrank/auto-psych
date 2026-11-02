# name: outlier_example_endorsement
# description: In 3-4 example sets where all but one example share a final digit, yes-rate when the target is the outlier example itself; observed below the null means people reject the off-pattern example more often than the model (which treats it as a near-certain member via the exception channel/noise), above means they accept it more.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    Dg = np.where(np.isnan(E), np.nan, np.mod(E, 10))
    is_ex = np.any(E == t[:, None], axis=1)
    same = np.sum(Dg[:, :, None] == Dg[:, None, :], axis=2).astype(float)
    same[np.isnan(Dg)] = np.nan
    s0 = np.nan_to_num(same, nan=0)
    is_out = (same == 1) & ((s0 == 1).sum(axis=1)[:, None] == 1) & (np.nanmax(same, axis=1)[:, None] == (n - 1)[:, None])
    valid = (n >= 3) & is_out.any(axis=1)
    out_val = np.where(valid, np.nansum(np.where(is_out, E, 0.0), axis=1), np.nan)
    maj_digit = np.nansum(np.where(~is_out & ~np.isnan(E), Dg, 0.0), axis=1) / np.maximum(n - 1, 1)
    d_out = np.abs(t - out_val)
    off = valid & (np.mod(t, 10) != maj_digit) & ~is_ex
    a = valid & (t == out_val)
    if a.sum() == 0:
        return 0.0
    return float(r[a].mean())
