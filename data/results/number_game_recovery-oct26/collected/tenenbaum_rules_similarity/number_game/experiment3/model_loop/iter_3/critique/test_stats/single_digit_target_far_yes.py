# name: single_digit_target_far_yes
# description: Yes-rate for single-digit non-example targets (1-9) when every example is >= 20; observed above null means people include small numbers more than the model, below null means people treat 'two-digit numbers' as a category boundary the model's ranges/rules over-cross.
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    dist = np.maximum(mn - t, t - mx)
    m = (mn >= 20) & (t <= 9) & ~is_ex
    return float(r[m].mean()) if m.any() else 0.0
