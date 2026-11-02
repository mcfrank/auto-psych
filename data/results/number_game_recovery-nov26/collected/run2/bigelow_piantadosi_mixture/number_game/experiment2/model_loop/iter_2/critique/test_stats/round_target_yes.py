# name: round_target_yes
# description: Yes rate for round targets (multiples of 10) that are not themselves examples; observed below null means people endorse round numbers less than the model predicts (model over-generalizes to round targets), above null the reverse.
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = np.sum(~np.isnan(X), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(X, axis=1)
    mx = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    m = (t % 10 == 0) & (dist > 0)
    return float(y[m].mean()) if m.sum() > 0 else 0.0
