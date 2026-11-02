# name: magnitude_scaled_near_extension
# description: Non-example targets lying outside the examples' [min,max] and 1-10 from the nearest example: yes-rate when that nearest example is >50 minus when it is <=50; observed above null_mean means people extend more generously around large numbers (log-like magnitude similarity) than the model, below means the model's magnitude effect is too strong.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    ex = sets.str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").values
    diff = np.abs(ex - t[:, None])
    diff = np.where(np.isnan(diff), 1e9, diff)
    j = np.argmin(diff, axis=1)
    dist = diff[np.arange(len(t)), j]
    near = ex[np.arange(len(t)), j]
    out = (t < np.nanmin(ex, axis=1)) | (t > np.nanmax(ex, axis=1))
    m = out & (dist >= 1) & (dist <= 10)
    r = df["rating"].values.astype(float)
    a = m & (near > 50); b = m & (near <= 50)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
