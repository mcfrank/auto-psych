# name: trial_index_yes_corr
# description: Correlation between trial_index (position in session) and yes response over non-example targets; observed below null means people say yes less as the session goes on (fatigue/conservatism drift absent from the model), above null means increasing yes bias.
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    ti = df["trial_index"].astype(float).values
    m = ~is_ex
    a = ti[m]
    y = r[m]
    if m.sum() < 3 or a.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(a, y)[0, 1])
