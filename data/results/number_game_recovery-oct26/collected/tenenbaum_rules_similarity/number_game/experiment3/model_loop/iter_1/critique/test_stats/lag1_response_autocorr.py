# name: lag1_response_autocorr
# description: Within-session lag-1 correlation of consecutive yes/no responses (consecutive trial_index); observed above null means responses perseverate (or drift in runs) beyond what the stimulus-driven model reproduces, below null means people alternate more than the model predicts.
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    last = X[np.arange(len(X)), n - 1]
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    ti = df["trial_index"].astype(int).values
    ok = np.diff(ti) == 1
    a = r[:-1][ok]
    b = r[1:][ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
