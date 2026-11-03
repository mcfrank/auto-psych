# name: distance_gradient_corr
# description: Pearson correlation between yes-response and log(1 + distance from target to nearest example); observed more negative than null means people show a steeper numeric-proximity gradient than the model's rules+intervals produce (less negative = model over-produces proximity).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    x = np.log1p(dist)
    y = df["rating"].astype(float).values
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
