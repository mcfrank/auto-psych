# name: distance_gradient_corr
# description: Correlation between rating and log(1 + distance from target to nearest example) over non-example targets; more negative than null means humans show a steeper numeric-similarity gradient than the model, less negative means a shallower one.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",").apply(lambda l: np.array([int(x) for x in l]))
    t = df["target"].astype(int).values
    dist = np.array([np.min(np.abs(a - x)) for a, x in zip(ex.values, t)])
    m = dist > 0
    x = np.log1p(dist[m]); y = df["rating"].values[m].astype(float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
