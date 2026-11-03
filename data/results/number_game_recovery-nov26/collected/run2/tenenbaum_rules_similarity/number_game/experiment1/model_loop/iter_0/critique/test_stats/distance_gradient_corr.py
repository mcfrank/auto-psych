# name: distance_gradient_corr
# description: Correlation between rating and log(1 + distance to nearest example) among non-member targets; observed more negative than null means humans show a steeper graded numeric-proximity falloff than the model's interval hypotheses give, less negative means flatter.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()[:, None].astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    member = np.any(valid & (ex == t), axis=1)
    dist = np.nanmin(np.where(valid, np.abs(ex - t), np.nan), axis=1)
    m = ~member
    x = np.log1p(dist[m]); y = r[m]
    if x.std() == 0 or y.std() == 0: return 0.0
    return float(np.corrcoef(x, y)[0, 1])
