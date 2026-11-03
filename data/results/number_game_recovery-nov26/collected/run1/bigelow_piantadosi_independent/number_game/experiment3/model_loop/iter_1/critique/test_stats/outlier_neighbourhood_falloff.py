# name: outlier_neighbourhood_falloff
# description: In 3-4 example sets where all but one example share a final digit, yes-rate for off-majority-digit non-example targets at distance 1 from the outlier example minus at distance 2-3; the model's radius prior (smaller radii likelier) predicts a positive falloff, so observed below the null means people endorse the outlier's neighbourhood more uniformly (flatter) than the model, above means a steeper falloff.
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
    a = off & (d_out == 1)
    b = off & (d_out >= 2) & (d_out <= 3)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
