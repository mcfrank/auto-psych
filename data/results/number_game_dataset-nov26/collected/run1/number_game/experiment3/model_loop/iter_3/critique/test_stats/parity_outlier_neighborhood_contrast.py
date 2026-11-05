# name: parity_outlier_neighborhood_contrast
# description: In 3-4-example sets where exactly one example differs in parity from the rest (e.g. 27 in "92,82,27,42"), yes-rate for targets within 10 of the outlier (and >10 from all other examples) minus yes-rate for targets within 10 of a majority example (and >10 from the outlier); observed < null means people discount the odd-one-out example more than the model's summed similarity does, observed > null means the model discounts it too much.
def test_statistic(df):
    s = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    v = s.values
    t = df["target"].astype(int).values
    n = (~np.isnan(v)).sum(axis=1)
    p = np.where(np.isnan(v), np.nan, v % 2)
    odd = np.nansum(p, axis=1)
    out_par = np.where(odd == 1, 1.0, np.where(odd == n - 1, 0.0, np.nan))
    has = (n >= 3) & ~np.isnan(out_par)
    is_out = (p == out_par[:, None]) & has[:, None]
    d = np.abs(v - t[:, None])
    d_out = np.where(is_out, d, np.inf).min(axis=1)
    d_maj = np.where(~is_out & ~np.isnan(v), d, np.inf).min(axis=1)
    r = df["rating"].values
    a = has & (d_out <= 10) & (d_maj > 10)
    b = has & (d_maj <= 10) & (d_out > 10)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
