# name: outlier_vs_majority_neighbour
# description: In 3-4 example sets where all but one example share a final digit, yes-rate for non-example targets within 3 of the outlier example minus yes-rate for off-digit non-example targets within 3 of a majority example (and not near the outlier); observed above the null means people favour the outlier's neighbourhood even more than the anchored-exception model, below means the model over-produces it.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    D = np.where(np.isnan(E), np.nan, np.mod(E, 10))
    # count of examples sharing each example's final digit
    same = np.sum(D[:, :, None] == D[:, None, :], axis=2).astype(float)
    same[np.isnan(D)] = np.nan
    is_out = (same == 1) & ((np.nan_to_num(same, nan=0) == 1).sum(axis=1)[:, None] == 1) & (np.nanmax(same, axis=1)[:, None] == (n - 1)[:, None])
    valid = (n >= 3) & is_out.any(axis=1)
    out_val = np.nansum(np.where(is_out, E, 0.0), axis=1)
    maj_digit = np.nansum(np.where(~is_out & ~np.isnan(E), D, 0.0), axis=1) / np.maximum(n - 1, 1)
    maj = np.where(~is_out & ~np.isnan(E), E, np.nan)
    d_out = np.abs(t - out_val)
    d_maj = np.nanmin(np.abs(maj - t[:, None]), axis=1) if valid.any() else np.full(len(t), np.inf)
    d_maj = np.where(np.isnan(d_maj), np.inf, d_maj)
    is_ex = np.any(E == t[:, None], axis=1)
    off = np.mod(t, 10) != maj_digit
    a = valid & ~is_ex & (d_out <= 3) & off
    b = valid & ~is_ex & (d_maj <= 3) & (d_out > 3) & off
    r = df["rating"].to_numpy().astype(float)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
