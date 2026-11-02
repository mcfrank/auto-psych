# name: distance_gradient_slope
# description: OLS slope of yes (0/1) on log(1 + distance from target to nearest example); more negative observed than null means people's similarity gradient is steeper than the model's, less negative means flatter.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()[:, None].astype(float)
    dist = np.where(~np.isnan(s), np.abs(s - t), np.inf).min(axis=1)
    x = np.log1p(dist)
    y = df["rating"].to_numpy(dtype=float)
    xc = x - x.mean()
    den = (xc ** 2).sum()
    if den == 0:
        return 0.0
    return float((xc * (y - y.mean())).sum() / den)
