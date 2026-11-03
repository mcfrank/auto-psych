# name: four_example_near_yes
# description: Yes rate for four-example sets at targets within 10 of some example (including inside the range); observed above null means people generalize more broadly from four examples than the size principle in the model allows.
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = np.sum(~np.isnan(X), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(X, axis=1)
    mx = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    m = (n == 4) & (dist <= 10)
    return float(y[m].mean()) if m.sum() > 0 else 0.0
