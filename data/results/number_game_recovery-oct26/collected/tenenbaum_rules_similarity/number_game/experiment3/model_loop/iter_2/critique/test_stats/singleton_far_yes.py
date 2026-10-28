# name: singleton_far_yes
# description: For single-example sets, yes-rate for targets more than 20 from the example; observed above null means people generalize broadly from one example (rule-based) more than the model, below null means the model over-produces far generalization from a single example.
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    m = (n == 1) & (np.abs(t - X[:, 0]) > 20)
    return float(r[m].mean()) if m.any() else 0.0
