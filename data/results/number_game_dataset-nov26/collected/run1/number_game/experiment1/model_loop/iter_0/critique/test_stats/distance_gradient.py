# name: distance_gradient
# description: Pearson correlation between yes and log(1 + distance from the target to the nearest example); observed more negative than null_mean means people show a stronger numeric-similarity (magnitude proximity) gradient than the model, less negative means weaker.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    t = df["target"].astype(float).values[:, None]
    d = np.nanmin(np.abs(ex - t), axis=1)
    x = np.log1p(d)
    y = df["rating"].astype(float).values
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
